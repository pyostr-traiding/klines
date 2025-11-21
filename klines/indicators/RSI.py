# klines/indicators/RSI.py

from typing import List, Union, Optional
from klines.base import _KlinesBase
from klines.schema.RSI import (
    RSISchema,
    StochRSISchema,
    PredictRSIResultSchema,
    PredictStochRSIResultSchema,
)


def simple_sma(values: list[float], period: int) -> float:
    if len(values) < period:
        raise ValueError("Недостаточно данных для SMA")
    return sum(values[-period:]) / period


class RSIIndicator(_KlinesBase):
    history_rsi: List[RSISchema] = []

    # ---------- RSI ----------
    def _calculate_RSI(
        self,
        closes: List[float],
        start_time: int,
        period: int = 14,
    ) -> RSISchema:
        gains: list[float] = []
        losses: list[float] = []

        for i in range(1, len(closes)):
            change = closes[i] - closes[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))

        if len(gains) < period:
            avg_gain = sum(gains) / period if gains else 0.0
            avg_loss = sum(losses) / period if losses else 0.0
        else:
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period
            for i in range(period, len(gains)):
                gain = gains[i]
                loss = losses[i]
                avg_gain = (avg_gain * (period - 1) + gain) / period
                avg_loss = (avg_loss * (period - 1) + loss) / period

        rsi_value = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
        return RSISchema(
            value=round(rsi_value, 2),
            kline_ms=start_time,
            interval=self.interval,
        )

    def current_RSI(self, period: int = 14) -> RSISchema:
        history = self.history[-period:]
        if not history:
            raise ValueError("Недостаточно истории для RSI")
        closes = [float(k.data[0].close) for k in history]
        start_time = int(history[-1].data[0].start)
        return self._calculate_RSI(closes, start_time, period)

    def history_RSI(self, period: int = 14) -> List[RSISchema]:
        rsi_values: list[RSISchema] = []
        total_candles = len(self.history)
        if total_candles < period:
            return rsi_values

        for i in range(total_candles - period + 1):
            window = self.history[i : i + period]
            closes = [float(k.data[0].close) for k in window]
            start_time = int(window[-1].data[0].start)
            rsi_values.append(self._calculate_RSI(closes, start_time, period))

        return rsi_values

    def load_history(self, period: int = 14) -> None:
        self.history_rsi = self.history_RSI(period=period)

    # ---------- STOCH RSI ----------
    def _calculate_stoch_RSI_from_history(
        self,
        history,
        rsi_period: int,
        stoch_period: int,
        k_period: int,
        d_period: int,
    ) -> StochRSISchema:
        min_history = rsi_period + stoch_period + k_period + d_period
        if len(history) < min_history:
            raise ValueError("Недостаточно истории для Stoch RSI")

        tail = history[-min_history:]
        rsi_series: list[float] = []

        for i in range(len(tail) - rsi_period + 1):
            sub = tail[i : i + rsi_period]
            closes = [float(k.data[0].close) for k in sub]
            start_time = int(sub[-1].data[0].start)
            rsi_series.append(self._calculate_RSI(closes, start_time, rsi_period).value)

        if len(rsi_series) < stoch_period + k_period + d_period - 1:
            raise ValueError("Недостаточно RSI значений")

        stoch_rsi_series: list[float] = []
        for i in range(len(rsi_series) - stoch_period + 1):
            window = rsi_series[i : i + stoch_period]
            rsi_now = window[-1]
            rsi_min = min(window)
            rsi_max = max(window)
            stoch_rsi_series.append(0.0 if rsi_max == rsi_min else (rsi_now - rsi_min) / (rsi_max - rsi_min) * 100)

        k = simple_sma(stoch_rsi_series, k_period)
        d_values = [
            simple_sma(stoch_rsi_series[i - k_period + 1 : i + 1], k_period)
            for i in range(k_period - 1, len(stoch_rsi_series))
        ]
        d = simple_sma(d_values, d_period)
        return StochRSISchema(value=[round(k, 2), round(d, 2)], kline_ms=int(tail[-1].data[0].start))

    def current_stoch_RSI(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
    ) -> StochRSISchema:
        min_history = rsi_period + stoch_period + k_period + d_period
        if len(self.history) < min_history:
            raise ValueError("Недостаточно данных в history для расчета Stoch RSI")
        return self._calculate_stoch_RSI_from_history(
            self.history,
            rsi_period=rsi_period,
            stoch_period=stoch_period,
            k_period=k_period,
            d_period=d_period,
        )

    # ---------- PREDICT RSI ----------
    def predict_rsi(
            self,
            side: str,
            rsi: float,
            target_range: float,  # теперь float, а не str
            period: int = 14,
            max_iter: int = 40,
    ) -> Union[PredictRSIResultSchema, None]:
        history = self.history[-period:]
        if not history:
            return None

        closes = [float(k.data[0].close) for k in history]
        start_time = int(history[-1].data[0].start)
        original_close = closes[-1]

        # Определяем диапазон для бинарного поиска
        lo, hi = (original_close * 0.5, original_close) if side == "buy" else (original_close, original_close * 1.5)
        result_close = result_rsi = None

        for _ in range(max_iter):
            mid = (lo + hi) / 2
            test_closes = closes[:-1] + [mid]
            test_rsi = self._calculate_RSI(test_closes, start_time, period).value

            # Проверяем попадание в target ± target_range
            if abs(test_rsi - rsi) <= target_range:
                result_close = mid
                result_rsi = test_rsi
                break

            if side == "buy":
                hi = mid if test_rsi > rsi + target_range else hi
                lo = lo if test_rsi > rsi + target_range else mid
            else:
                lo = mid if test_rsi < rsi - target_range else lo
                hi = hi if test_rsi < rsi - target_range else mid

        if result_close is None:
            return None

        delta_percent = ((result_close - original_close) / original_close) * 100
        return PredictRSIResultSchema(
            side=side,
            rate=round(result_close, 2),
            percent=round(delta_percent, 2),
            kline_ms=start_time,
            interval=self.interval,
            rsi=round(result_rsi if result_rsi is not None else rsi, 2),
        )

    # ---------- PREDICT STOCH RSI ----------
    def predict_stoch_rsi(
            self,
            side: str,
            target_range: float,  # float вместо строки
            rsi_period: int = 14,
            stoch_period: int = 14,
            k_period: int = 3,
            d_period: int = 3,
            max_iter: int = 40,
    ) -> Union[PredictStochRSIResultSchema, None]:
        min_history = rsi_period + stoch_period + k_period + d_period
        if len(self.history) < min_history:
            raise ValueError("Недостаточно истории для предсказания Stochastic RSI")

        last_kline = self.history[-1]
        original_close = float(last_kline.data[0].close)
        lo, hi = (original_close * 0.1, original_close) if side == "buy" else (original_close, original_close * 0.1)

        result_close = result_k = result_d = result_ms = None

        try:
            for _ in range(max_iter):
                mid = (lo + hi) / 2
                last_kline.data[0].close = mid

                stoch_rsi = self._calculate_stoch_RSI_from_history(
                    self.history,
                    rsi_period=rsi_period,
                    stoch_period=stoch_period,
                    k_period=k_period,
                    d_period=d_period,
                )
                k_val, d_val = stoch_rsi.value

                # Попадание в target ± target_range
                if abs(k_val - 50) <= target_range:  # 50 здесь можно заменить на текущее k_val, если нужно
                    result_close = mid
                    result_k = k_val
                    result_d = d_val
                    result_ms = stoch_rsi.kline_ms
                    break

                if side == "buy":
                    hi = mid if k_val > 50 + target_range else hi
                    lo = lo if k_val > 50 + target_range else mid
                else:
                    lo = mid if k_val < 50 - target_range else lo
                    hi = hi if k_val < 50 - target_range else mid
        finally:
            last_kline.data[0].close = original_close

        if result_close is None:
            return None

        delta_percent = ((result_close - original_close) / original_close) * 100
        return PredictStochRSIResultSchema(
            side=side,
            rate=round(result_close, 2),
            percent=round(delta_percent, 2),
            kline_ms=result_ms,
            interval=self.interval,
            k=round(result_k, 2),
            d=round(result_d, 2),
        )

