import json
from typing import List, Generator

from dotenv import load_dotenv
from redis import Redis

from klines.schema.kline import KlineSchema, CandleSchema
from klines.utils import ms_to_dt


def list_to_schema(interval: int, symbol: str, data) -> List[KlineSchema]:
    """
    Преобразовать список свечей (dict) в список KlineSchema.
    Возвращает в порядке старая -> новая.
    """
    result: List[KlineSchema] = []
    for k in reversed(data):
        if isinstance(k, KlineSchema):
            result.append(k)
            continue
        kline = KlineSchema(
            topic=f'kline.{interval}.{symbol}',
            symbol=symbol,
            interval=interval,
            data=[CandleSchema(
                start=k['ts'],
                interval=interval,
                open=k['o'],
                close=k['c'],
                high=k['h'],
                low=k['l'],
                volume=k['v'],
                turnover=k['t'],
                dt=k['dt']
            )],
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

    def __init__(self, symbol: str, interval: int, exchange: str, redis_candles: Redis, end: int = None, max_length: int = 1000):
        self.max_length = max_length
        self.symbol = symbol
        self.interval = interval
        self.exchange = exchange
        self.end = end
        self.redis_candles = redis_candles

        self.history: List[KlineSchema] = self._get_history()
        if not self.history:
            raise RuntimeError('Empty history')

        self._check_and_trim_history()
        self._check_full_sequence_once()
        print(f'[{self.symbol} {self.interval}] История загружена [{len(self.history)} свечей]')

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

    def _get_history(self) -> List[KlineSchema]:
        load_dotenv()
        key = f'candles:{self.symbol}:{self.interval}:{self.exchange}'
        res = self.redis_candles.zrevrange(key, 0, self.max_length - 1)
        if not res:
            return []
        klines = [json.loads(i) for i in res]
        return list_to_schema(interval=self.interval, symbol=self.symbol, data=klines)

    def _check_full_sequence_once(self) -> bool:
        expected_diff = self.interval * 60 * 1000
        ok = True
        for i in range(1, len(self.history)):
            prev = self.history[i - 1].data[0].start
            curr = self.history[i].data[0].start
            if curr - prev != expected_diff:
                ok = False
        return ok

    def _check_tail_sequence(self) -> bool:
        if len(self.history) < 2:
            return True
        expected_diff = self.interval * 60 * 1000
        prev = self.history[-2].data[0].start
        curr = self.history[-1].data[0].start
        return curr - prev == expected_diff

    def is_valid_kline_to_history(self, kline: KlineSchema) -> bool:
        return kline.symbol == self.symbol and kline.interval == self.interval

    def _check_and_trim_history(self) -> bool:
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

    def _update(self, kline: KlineSchema) -> bool:
        self.history[-1] = kline
        self.last_kline = kline.data[0]
        self.end = kline.data[0].start
        return True

    def _add(self, kline: KlineSchema) -> bool:
        self.history.append(kline)
        self._check_and_trim_history()
        self._check_tail_sequence()
        return True

    def update(self, kline: KlineSchema) -> bool:
        if not self.is_valid_kline_to_history(kline):
            return False
        k_start = kline.data[0].start
        last_start = self.last_kline.start
        if k_start == last_start:
            return self._update(kline)
        if k_start > last_start:
            return self._add(kline)
        return False
