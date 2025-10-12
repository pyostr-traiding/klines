import os
import json

from typing import List

import redis
from dotenv import load_dotenv

from schema.kline import KlineSchema, CandleSchema
from .utils import ms_to_dt


def list_to_schema(
        interval,
        symbol,
        data,
):
    """
    Преобразовать список свечей в список схем
    """
    return [
        KlineSchema(
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
                    dt=k['dt']
                )
            ],
        )
        for k in reversed(data)
    ]

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

    def __init__(
            self,
            symbol: str,
            interval: int,
            exchange: str,
            end: int = None,
            max_length: int = 1000,
    ):
        self.max_length = max_length
        self.symbol = symbol
        self.interval = interval
        self.exchange = exchange
        self.end = end

        print(f'[{symbol} {interval}] Загрузка данный')
        self.history = self._get_history()
        print(f'[{symbol} {interval}] История загружена [{len(self.history)} свечей]')
        self.check_and_trim_history()
        self.length = len(self.history)
        self.last_kline = self.history[-1].data[0]
        self.check_kline_sequence()

    @property
    def start_str(self) -> str:
        """
        Возвращает start в формате: timestamp | дата-время
        """
        dt = ms_to_dt(self.start)
        return f"{self.start} | {dt}"

    @property
    def end_str(self) -> str:
        """
        Возвращает end в формате: timestamp | дата-время
        """
        if self.end is None:
            return self.history[-1].data[0].start_str
        dt = ms_to_dt(self.end)
        return f"{self.end} | {dt}"


    def _get_history(self,) -> List[KlineSchema]:
        """
        Возвращает список свечей в порядке старая -> новая
        """
        load_dotenv()
        server_redis = redis.Redis(
            host=os.getenv('REDIS_HOST'),
            port=int(os.getenv('REDIS_PORT')),
            password=os.getenv('REDIS_PASSWORD'),
            db=0,
            decode_responses=True
        )
        key = f'candles:{self.symbol}:{self.interval}:{self.exchange}'
        res = server_redis.zrevrange(key, 0, 999)  # с конца (самые новые)
        if not res:
            print('Ошибка получения свечей истории')
            exit()
        klines = [json.loads(i) for i in res]
        return list_to_schema(
            interval=self.interval,
            symbol=self.symbol,
            data=klines
        )

    def check_kline_sequence(self) -> bool:
        """
        Проверим что все свечи последовательны
        Нужно что бы убедиться в корректности истории
        Возможно потеря соединения/баг и прочие условия вызывающие
        нарушение интервала свечей
        """
        expected_diff = self.interval * 60 * 1000

        for i in range(1, len(self.history)):
            prev = self.history[i - 1].data[0].start
            curr = self.history[i].data[0].start
            if curr - prev != expected_diff:
                print('Последовательность нарушена!')
                history = self._get_history()

                self.history = list_to_schema(
                    interval=self.interval,
                    symbol=self.symbol,
                    data=history
                )
                print('Последовательность восстановлена!')
                self.check_kline_sequence()
                return True
        return True

    def is_valid_kline_to_history(
            self,
            kline: KlineSchema,
    ) -> bool:
        """
        Проверяет что свеча относится к этой истории
        """
        if kline.symbol != self.symbol:
            print(f'Свеча не того символа: {kline.symbol} != {self.symbol}')
            return False
        if kline.interval != self.interval:
            print(f'Свеча не того интервала: {kline.interval} != {self.interval}')
            return False
        return True

    def check_and_trim_history(
            self,
    ):
        """
        Обрезание истории по максимальной длине

        Проверяем длину истории

        Если длина превышена сокращаем историю
        """
        history_length = len(self.history)
        if history_length > self.max_length:
            self.history = self.history[-self.max_length:]
            self.start = self.history[0].data[0].start
        return True

    def _update(
            self,
            kline: KlineSchema,
    ) -> bool:
        """
        Обновляем значение
        """
        self.history[-1] = kline
        self.end = kline.data[0].start
        return True

    def _add(
            self,
            kline: KlineSchema,
    ) -> bool:
        """
        Добавляем новую свечу
        """
        # Добавляем в конец
        self.history.append(kline)
        self.check_and_trim_history()

        # Устанавливаем новое время актуальной свечи
        self.end = self.history[-1].data[0].start

        # Проверяем валидность истории
        self.check_kline_sequence()
        return True

    def update(
            self,
            kline: KlineSchema,
    ) -> bool:
        """
        Сюда отправляем обновления свечей

        Если свеча уже есть в истории - обновляем ее
        Если свеча новая - добавляем в конец
        Так же проверяем длину истории
        """
        if not self.is_valid_kline_to_history(kline):
            return False
        # Если свеча есть - обновляем
        if kline.data[0].start == self.last_kline.start:
            # заменяем последнюю свечу
            if self._update(kline):
                self.last_kline = kline.data[-1]
                return True
        # Если свечи нет
        if kline.data[0].start != self.last_kline.start:
            if self._add(kline):
                self.last_kline = kline.data[-1]
                return True
