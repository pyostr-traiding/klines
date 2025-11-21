# klines/indicators/RSI.py

from typing import List, Union

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
    rsi: RSISchema
    stoch_rsi: StochRSISchema

    # ---------- RSI ----------

    def _calculate_RSI(
        self,
        closes: List[float],
        start_time: int,
        period: int = 14,
    ) -> RSISchema:
        """
        Рассчитывает RSI по списку цен закрытия.
        """
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

        if avg_loss == 0:
            rsi_value = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_value = 100 - (100 / (1 + rs))

        return RSISchema(
            value=round(rsi_value, 2),
            kline_ms=start_time,
            interval=self.interval,
        )

    def current_RSI(
        self,
        period: int = 14,
    ) -> RSISchema:
        """
        Текущее значение RSI.
        """
        history = self.history[-period:]
        if not history:
            raise ValueError("Недостаточно истории для RSI")

        close_prices = [float(i.data[0].close) for i in history]
        start_time = int(history[-1].data[0].start)

        return self._calculate_RSI(close_prices, start_time, period)

    def history_RSI(
        self,
        period: int = 14,
    ) -> List[RSISchema]:
        """
        История RSI по каждой свече.
        """
        rsi_values: list[RSISchema] = []
        total_candles = len(self.history)

        if total_candles < period:
            return rsi_values

        for i in range(total_candles - period + 1):
            window = self.history[i : i + period]
            closes = [float(kline.data[0].close) for kline in window]
            start_time = int(window[-1].data[0].start)
            rsi = self._calculate_RSI(closes, start_time, period)
            rsi_values.append(rsi)

        return rsi_values

    def load_history(
        self,
        period: int = 14,
    ) -> None:
        """
        Загрузка истории индикатора
        """
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
        """
        Расчёт Stochastic RSI по хвосту истории.
        Берём только минимально необходимый кусок history.
        """
        min_history = rsi_period + stoch_period + k_period + d_period
        if len(history) < min_history:
            raise ValueError("Недостаточно истории для Stoch RSI")

        tail = history[-min_history:]

        rsi_series: list[float] = []
        for i in range(len(tail) - rsi_period + 1):
            sub_history = tail[i : i + rsi_period]
            closes = [float(kline.data[0].close) for kline in sub_history]
            start_time = int(sub_history[-1].data[0].start)
            rsi = self._calculate_RSI(closes, start_time, rsi_period)
            rsi_series.append(rsi.value)

        if len(rsi_series) < stoch_period + k_period + d_period - 1:
            raise ValueError("Недостаточно RSI значений")

        stoch_rsi_series: list[float] = []
        for i in range(len(rsi_series) - stoch_period + 1):
            window = rsi_series[i : i + stoch_period]
            rsi_now = window[-1]
            rsi_min = min(window)
            rsi_max = max(window)
            if rsi_max == rsi_min:
                stoch = 0.0
            else:
                stoch = (rsi_now - rsi_min) / (rsi_max - rsi_min)
            stoch_rsi_series.append(stoch * 100)

        if len(stoch_rsi_series) < k_period:
            raise ValueError("Недостаточно stoch_rsi значений для сглаживания K")

        k = simple_sma(stoch_rsi_series, k_period)

        d_values = [
            simple_sma(stoch_rsi_series[i - k_period + 1 : i + 1], k_period)
            for i in range(k_period - 1, len(stoch_rsi_series))
        ]
        if len(d_values) < d_period:
            raise ValueError("Недостаточно сглаженных K значений для расчёта D")

        d = simple_sma(d_values, d_period)

        last_kline = tail[-1]
        start_time = int(last_kline.data[0].start)

        return StochRSISchema(
            value=[round(k, 2), round(d, 2)],
            kline_ms=start_time,
        )

    def current_stoch_RSI(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
    ) -> StochRSISchema:
        """
        Текущее значение Stochastic RSI (%K, %D)
        """
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
        target_range: str,
        period: int = 14,
        max_iter: int = 40,
    ) -> Union[PredictRSIResultSchema, None]:
        """
        Подбор значения close, при котором RSI попадет в указанный диапазон.
        Бинарный поиск по цене на последнем отрезке history[-period:].
        """
        history = self.history[-period:]
        if not history:
            return None

        closes = [float(i.data[0].close) for i in history]
        start_time = int(history[-1].data[0].start)

        from_lev, to_lev = map(float, target_range.split("-"))
        original_close = closes[-1]

        if side == "buy":
            lo = original_close * 0.5
            hi = original_close
        else:
            lo = original_close
            hi = original_close * 1.5

        result_close = None
        result_rsi = None

        for _ in range(max_iter):
            mid = (lo + hi) / 2
            test_closes = closes[:-1] + [mid]

            test_rsi = self._calculate_RSI(
                test_closes,
                start_time,
                period,
            ).value

            if from_lev <= test_rsi <= to_lev:
                result_close = mid
                result_rsi = test_rsi
                break

            if side == "buy":
                if test_rsi > to_lev:
                    hi = mid
                else:
                    lo = mid
            else:
                if test_rsi < from_lev:
                    lo = mid
                else:
                    hi = mid

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
        target_range: str,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
        max_iter: int = 40,
    ) -> Union[PredictStochRSIResultSchema, None]:
        """
        Подбор close, при котором %K попадёт в указанный диапазон.
        Бинарный поиск, перерасчёт Stoch RSI только по хвосту.
        """
        min_history = rsi_period + stoch_period + k_period + d_period
        if len(self.history) < min_history:
            raise ValueError("Недостаточно истории для предсказания Stochastic RSI")

        from_lev, to_lev = map(float, target_range.split("-"))

        last_kline = self.history[-1]
        original_close = float(last_kline.data[0].close)

        if side == "buy":
            lo = original_close * 0.5
            hi = original_close
        else:
            lo = original_close
            hi = original_close * 1.5

        result_close = None
        result_k = None
        result_d = None
        result_ms = None

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

                k_value = stoch_rsi.value[0]
                d_value = stoch_rsi.value[1]

                if from_lev <= k_value <= to_lev:
                    result_close = mid
                    result_k = k_value
                    result_d = d_value
                    result_ms = stoch_rsi.kline_ms
                    break

                if side == "buy":
                    if k_value > to_lev:
                        hi = mid
                    else:
                        lo = mid
                else:
                    if k_value < from_lev:
                        lo = mid
                    else:
                        hi = mid
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
