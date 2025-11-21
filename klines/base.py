import json
from typing import List

from dotenv import load_dotenv
from redis import Redis

from klines.schema.kline import KlineSchema, CandleSchema
from klines.utils import ms_to_dt


def list_to_schema(
        interval: int,
        symbol: str,
        data,
) -> List[KlineSchema]:
    """
    Преобразовать список свечей (dict) в список KlineSchema.
    Возвращает в порядке старая -> новая.
    """
    result: List[KlineSchema] = []
    for k in reversed(data):
        # если вдруг уже KlineSchema — пропускаем
        if isinstance(k, KlineSchema):
            result.append(k)
            continue
        kline = KlineSchema(
            topic=f'kline.{interval}.{symbol}',
            symbol=symbol,
            interval=interval,
            data=[
                CandleSchema(
                    start=k['ts'],
                    interval=interval,
                    open=k['o'],
                    close=k['c'],
                    high=k['h'],
                    low=k['l'],
                    volume=k['v'],
                    turnover=k['t'],
                    dt=k['dt'],
                )
            ],
        )
        result.append(kline)
    return result


class _KlinesBase:
    max_length: int
    history: List[KlineSchema]
    interval: int
    start: int
    end: int
    symbol: str
    length: int
    exchange: str

    last_kline: CandleSchema
    redis_candles: Redis

    def __init__(
            self,
            symbol: str,
            interval: int,
            exchange: str,
            redis_candles: Redis,
            end: int = None,
            max_length: int = 1000,
    ):
        self.max_length = max_length
        self.symbol = symbol
        self.interval = interval
        self.exchange = exchange
        self.end = end
        self.redis_candles = redis_candles

        self.history: List[KlineSchema] = self._get_history()
        if not self.history:
            print('Ошибка получения свечей истории')
            raise RuntimeError('Empty history')

        # Подрезаем историю и синхронизируем метаданные
        self._check_and_trim_history()

        # Однократная проверка последовательности при старте (по всей истории)
        self._check_full_sequence_once()

        print(f'[{self.symbol} {self.interval}] История загружена [{len(self.history)} свечей]')

    # -------------------------------------------------------------------------
    # Свойства для удобного отображения
    # -------------------------------------------------------------------------
    @property
    def start_str(self) -> str:
        dt = ms_to_dt(self.start)
        return f"{self.start} | {dt}"

    @property
    def end_str(self) -> str:
        if self.end is None and self.history:
            return self.history[-1].data[0].start_str
        dt = ms_to_dt(self.end)
        return f"{self.end} | {dt}"

    # -------------------------------------------------------------------------
    # Загрузка истории
    # -------------------------------------------------------------------------
    def _get_history(self) -> List[KlineSchema]:
        """
        Возвращает список свечей (старая -> новая) из Redis.
        """
        load_dotenv()
        key = f'candles:{self.symbol}:{self.interval}:{self.exchange}'
        # самые новые сначала
        res = self.redis_candles.zrevrange(key, 0, self.max_length - 1)
        if not res:
            return []
        klines = [json.loads(i) for i in res]
        return list_to_schema(
            interval=self.interval,
            symbol=self.symbol,
            data=klines,
        )

    # -------------------------------------------------------------------------
    # Проверка полной последовательности (ТОЛЬКО при старте)
    # -------------------------------------------------------------------------
    def _check_full_sequence_once(self) -> bool:
        """
        Однократная проверка корректности истории при инициализации.
        Без рекурсии и перезагрузок.
        """
        expected_diff = self.interval * 60 * 1000
        ok = True

        for i in range(1, len(self.history)):
            prev = self.history[i - 1].data[0].start
            curr = self.history[i].data[0].start
            if curr - prev != expected_diff:
                print(
                    f'Предупреждение: последовательность нарушена '
                    f'между индексами {i - 1} и {i}: {curr - prev} != {expected_diff}'
                )
                ok = False
                # дальше можно либо продолжить, либо выйти — оставим продолжение
        return ok

    # -------------------------------------------------------------------------
    # Локальная проверка последовательности по последним двум свечам
    # -------------------------------------------------------------------------
    def _check_tail_sequence(self) -> bool:
        """
        Быстрая проверка последовательности по последним двум свечам.
        Используется только при добавлении новой свечи.
        """
        if len(self.history) < 2:
            return True

        expected_diff = self.interval * 60 * 1000
        prev = self.history[-2].data[0].start
        curr = self.history[-1].data[0].start

        if curr - prev != expected_diff:
            print(
                f'Предупреждение: последовательность нарушена по последней свече: '
                f'{curr - prev} != {expected_diff} ({prev} -> {curr})'
            )
            return False
        return True

    # -------------------------------------------------------------------------
    # Валидация свечи
    # -------------------------------------------------------------------------
    def is_valid_kline_to_history(
            self,
            kline: KlineSchema,
    ) -> bool:
        if kline.symbol != self.symbol:
            print(f'Свеча не того символа: {kline.symbol} != {self.symbol}')
            return False
        if kline.interval != self.interval:
            print(f'Свеча не того интервала: {kline.interval} != {self.interval}')
            return False
        return True

    # -------------------------------------------------------------------------
    # Обрезка истории и синхронизация метаданных
    # -------------------------------------------------------------------------
    def _check_and_trim_history(self) -> bool:
        """
        Подрезает историю по max_length и обновляет start/end/last_kline/length.
        """
        history_length = len(self.history)
        if history_length == 0:
            return False

        if history_length > self.max_length:
            self.history = self.history[-self.max_length:]
            history_length = len(self.history)

        self.start = self.history[0].data[0].start
        self.end = self.history[-1].data[0].start
        self.last_kline = self.history[-1].data[0]
        self.length = history_length
        return True

    # -------------------------------------------------------------------------
    # Внутреннее обновление последней свечи
    # -------------------------------------------------------------------------
    def _update(
            self,
            kline: KlineSchema,
    ) -> bool:
        """
        Обновляет последнюю свечу (bybit kline.update).
        """
        self.history[-1] = kline
        self.last_kline = kline.data[0]
        self.end = kline.data[0].start
        # длина не меняется
        return True

    # -------------------------------------------------------------------------
    # Внутреннее добавление новой свечи
    # -------------------------------------------------------------------------
    def _add(
            self,
            kline: KlineSchema,
    ) -> bool:
        """
        Добавляет новую свечу в конец (закрытие старой + открытие новой).
        """
        self.history.append(kline)
        # подрезаем историю + обновляем start/end/last_kline/length
        self._check_and_trim_history()
        # проверяем только последнюю пару
        self._check_tail_sequence()
        return True

    # -------------------------------------------------------------------------
    # Публичный метод обновления истории
    # -------------------------------------------------------------------------
    def update(
            self,
            kline: KlineSchema,
    ) -> bool:
        """
        Принимает обновление свечей from bybit kline.update:

        - если start совпадает с последней свечой — обновляем её
        - если start больше — добавляем новую свечу в конец
        """
        if not self.is_valid_kline_to_history(kline):
            return False

        k_start = kline.data[0].start
        last_start = self.last_kline.start

        # обновление текущей (незакрывшейся) свечи
        if k_start == last_start:
            return self._update(kline)

        # новая свеча
        if k_start > last_start:
            return self._add(kline)

        # историческое обновление или мусор — игнорируем
        # (bybit update обычно не шлёт "старые" свечи, но на всякий случай)
        print(
            f'Предупреждение: получена свеча с прошедшим временем '
            f'{k_start} < {last_start}, игнор.'
        )
        return False
