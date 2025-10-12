from typing import List
import pandas as pd

from indicators.base import _KlinesBase
from indicators.schema.MACD import MACDSchema


class MACDIndicator(_KlinesBase):
    @staticmethod
    def _calculate_MACD(
            closes: List[float],
            start_time: int
    ) -> MACDSchema:
        """
        Вычисляет MACD, сигнальную линию и гистограмму.
        """
        df = pd.DataFrame(closes, columns=['close'])

        # Параметры
        span_12 = 12
        span_26 = 26
        span_signal = 9

        # EMA
        df['EMA_12'] = df['close'].ewm(span=span_12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=span_26, adjust=False).mean()

        # MACD и Signal
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=span_signal, adjust=True).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

        return MACDSchema(
            hist=round(df['MACD_Histogram'].iloc[-1], 2),
            macd=round(df['MACD'].iloc[-1], 2),
            sign=round(df['Signal_Line'].iloc[-1], 2),
            kline_ms=start_time
        )

    def current_MACD(
            self,
            period: int = 26,
    ):
        """
        Текущее значение MACD.
        """
        history = self.history[-period + 1:]
        close_prices = [float(i.data[0].close) for i in history]
        start_time = int(history[-1].data[0].start)  # Берём время открытия самой первой свечи

        return self._calculate_MACD(close_prices, start_time)

    def history_MACD(
            self,
            period: int = 30,
    ) -> List[MACDSchema]:
        """
        Исторические значения MACD.
        """
        macd_values = []
        total_candles = len(self.history)

        if total_candles < period:
            return macd_values

        for i in range(total_candles - period + 1):
            window = self.history[i:i + period]
            closes = [float(kline.data[0].close) for kline in window]
            start_time = int(window[-1].data[0].start)  # Время открытия первой свечи окна

            macd = self._calculate_MACD(closes, start_time)
            macd_values.append(macd)

        return macd_values
