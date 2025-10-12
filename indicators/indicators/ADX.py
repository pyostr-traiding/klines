from typing import List, Literal

from indicators.base import _KlinesBase
from indicators.schema.ADX import ADXSchema
from indicators.schema.kline import CandleSchema

Mode = Literal["simple", "wilder", "ema"]


class ADXIndicator(_KlinesBase):
    @staticmethod
    def __true_range(high: float, low: float, prev_close: float) -> float:
        return max(high - low, abs(high - prev_close), abs(low - prev_close))

    @staticmethod
    def __directional_movement(high: float, low: float, prev_high: float, prev_low: float):
        plus_dm = high - prev_high if (high - prev_high) > (prev_low - low) and (high - prev_high) > 0 else 0.0
        minus_dm = prev_low - low if (prev_low - low) > (high - prev_high) and (prev_low - low) > 0 else 0.0
        return plus_dm, minus_dm

    @staticmethod
    def __ema(values: List[float], period: int) -> List[float]:
        if len(values) < period:
            return []
        k = 2 / (period + 1)
        ema_values = [sum(values[:period]) / period]
        for v in values[period:]:
            ema_values.append((v - ema_values[-1]) * k + ema_values[-1])
        return ema_values

    @staticmethod
    def __wilder_smoothing(values: List[float], period: int) -> List[float]:
        if len(values) < period:
            return []
        smoothed = [sum(values[:period])]
        for v in values[period:]:
            smoothed.append(smoothed[-1] - (smoothed[-1] / period) + v)
        return smoothed

    def _calculate_ADX(
            self,
            kline_list: List[CandleSchema],
            period: int = 14,
            mode: Mode = "wilder"
    ) -> float:
        if len(kline_list) < period + 1:
            return 0.0

        highs = [float(c.high) for c in kline_list]
        lows = [float(c.low) for c in kline_list]
        closes = [float(c.close) for c in kline_list]

        tr_list, plus_dm_list, minus_dm_list = [], [], []

        for i in range(1, len(kline_list)):
            tr = self.__true_range(highs[i], lows[i], closes[i - 1])
            plus_dm, minus_dm = self.__directional_movement(highs[i], lows[i], highs[i - 1], lows[i - 1])

            tr_list.append(tr)
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)

        if mode == "wilder":
            if len(kline_list) < 2 * period:
                return 0.0
            tr_sm = self.__wilder_smoothing(tr_list, period)
            plus_sm = self.__wilder_smoothing(plus_dm_list, period)
            minus_sm = self.__wilder_smoothing(minus_dm_list, period)
        elif mode == "ema":
            tr_sm = self.__ema(tr_list, period)
            plus_sm = self.__ema(plus_dm_list, period)
            minus_sm = self.__ema(minus_dm_list, period)
        elif mode == "simple":
            tr_sm = [sum(tr_list[-period:])]
            plus_sm = [sum(plus_dm_list[-period:])]
            minus_sm = [sum(minus_dm_list[-period:])]
        else:
            raise ValueError("Неверный режим сглаживания")

        dx_list = []
        for tr, pdm, mdm in zip(tr_sm, plus_sm, minus_sm):
            if tr == 0:
                dx_list.append(0)
                continue
            plus_di = 100 * (pdm / tr)
            minus_di = 100 * (mdm / tr)
            di_sum = plus_di + minus_di
            dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum != 0 else 0
            dx_list.append(dx)

        if mode == "simple":
            return round(dx_list[-1], 2)

        if len(dx_list) < period:
            return 0.0

        if mode == "wilder":
            adx = [sum(dx_list[:period]) / period]
            for dx in dx_list[period:]:
                adx.append(((adx[-1] * (period - 1)) + dx) / period)
        elif mode == "ema":
            adx = self.__ema(dx_list, period)
        else:
            adx = []

        return round(adx[-1], 2) if adx else 0.0

    def current_ADX(
            self,
            period: int = 14,
    ) -> ADXSchema:
        """
        Получить текущее значение ADX + время начала окна.
        """
        candles = [i.data[0] for i in self.history]
        if len(candles) < 2 * period:
            return ADXSchema(simple=0.0, wilder=0.0, ema=0.0, kline_ms=0)

        candles_window = candles[-(2 * period):]  # для wilder нужно больше данных
        start_time = candles_window[-1].start

        adx_simple = self._calculate_ADX(candles_window, period=period, mode="simple")
        adx_wilder = self._calculate_ADX(candles_window, period=period, mode="wilder")
        adx_ema = self._calculate_ADX(candles_window, period=period, mode="ema")

        return ADXSchema(
            simple=adx_simple,
            wilder=adx_wilder,
            ema=adx_ema,
            kline_ms=start_time,
        )

    def history_ADX(
            self,
            period: int = 15,
    ) -> List[ADXSchema]:
        """
        История ADX по каждой свече.
        """
        result = []
        candles = [i.data[0] for i in self.history]
        total = len(candles)
        min_len = 2 * period  # для Wilder

        if total < min_len:
            return result

        for i in range(total - min_len + 1):
            window = candles[i:i + min_len]
            start_time = window[-1].start

            adx_simple = self._calculate_ADX(window, period=period, mode="simple")
            adx_wilder = self._calculate_ADX(window, period=period, mode="wilder")
            adx_ema = self._calculate_ADX(window, period=period, mode="ema")

            result.append(ADXSchema(
                simple=adx_simple,
                wilder=adx_wilder,
                ema=adx_ema,
                kline_ms=start_time,
            ))

        return result
