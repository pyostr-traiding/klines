from typing import List, Union

from klines.base import _KlinesBase
from klines.schema.RSI import RSISchema, StochRSISchema, PredictRSIResultSchema, PredictStochRSIResultSchema


def simple_sma(values: list[float], period: int) -> float:
    if len(values) < period:
        print(len(values), period)
        raise ValueError("Недостаточно данных для SMA")
    return sum(values[-period:]) / period


class RSIIndicator(_KlinesBase):
    history_rsi: List[RSISchema] = []
    rsi: RSISchema
    stoch_rsi: StochRSISchema

    def _calculate_RSI(
            self,
            closes: List[float],
            start_time: int,
            period: int = 14,
    ) -> RSISchema:
        """
        Рассчитывает RSI для списка цен закрытия с использованием SMA для сглаживания.
        """
        gains = []
        losses = []

        # Вычисление изменений для всех свечей
        for i in range(1, len(closes)):
            change = closes[i] - closes[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        # Вычисляем первые значения для SMA
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Применяем SMA для сглаживания последующих значений
        for i in range(period, len(closes)):
            gain = gains[i]
            loss = losses[i]

            # Сглаживаем средние значения с помощью SMA
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        # Если потерь нет, RSI будет 100, иначе считаем RSI
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        return RSISchema(
            value=round(rsi, 2),
            kline_ms=start_time,
            interval=self.interval
        )

    def current_RSI(
            self,
            period: int = 14,
    ) -> RSISchema:
        """
        Получить текущее значение RSI.
        """
        history = self.history[-period:]
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
        rsi_values = []
        total_candles = len(self.history)

        if total_candles < period:
            return rsi_values

        for i in range(total_candles - period + 1):
            window = self.history[i:i + period]
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
        history = self.history_RSI(
            period=period,
        )
        self.history_rsi = history

    def current_stoch_RSI(
            self,
            rsi_period: int = 14,
            stoch_period: int = 14,
            k_period: int = 3,
            d_period: int = 3
    ) -> StochRSISchema:
        """
        Рассчитывает Stochastic RSI (%K, %D) с параметрами (rsi_period, stoch_period, k_period, d_period)
        Возвращает кортеж (%K, %D), как на TradingView/Bybit.
        """

        min_history = rsi_period + stoch_period + k_period + d_period
        if len(self.history) < min_history:
            raise ValueError("Недостаточно данных в history для расчета Stoch RSI")

        # Вычисляем серию RSI значений
        rsi_series = []
        for i in range(len(self.history) - rsi_period + 1):
            sub_history = self.history[i:i + rsi_period]
            close_prices = [float(kline.data[0].close) for kline in sub_history]
            start_time = int(sub_history[-1].data[0].start)
            rsi = self._calculate_RSI(close_prices, start_time, rsi_period)
            rsi_series.append(rsi.value)

        if len(rsi_series) < stoch_period + k_period + d_period - 1:
            raise ValueError("Недостаточно RSI значений для расчета Stoch RSI")

        # Вычисляем Stoch RSI по RSI серии
        stoch_rsi_series = []
        for i in range(len(rsi_series) - stoch_period + 1):
            window = rsi_series[i:i + stoch_period]
            rsi_now = window[-1]
            rsi_min = min(window)
            rsi_max = max(window)
            if rsi_max == rsi_min:
                stoch = 0.0
            else:
                stoch = (rsi_now - rsi_min) / (rsi_max - rsi_min)
            stoch_rsi_series.append(stoch * 100)  # в процентах, как на бирже

        if len(stoch_rsi_series) < k_period:
            raise ValueError("Недостаточно stoch_rsi значений для сглаживания K")

        # Сглаженное значение K
        k = simple_sma(stoch_rsi_series, k_period)

        # Строим серию сглаженных K значений для вычисления D
        d_values = [
            simple_sma(stoch_rsi_series[i - k_period + 1:i + 1], k_period)
            for i in range(k_period - 1, len(stoch_rsi_series))
        ]

        if len(d_values) < d_period:
            raise ValueError("Недостаточно сглаженных K значений для расчета D")

        # Сглаженное значение D
        d = simple_sma(d_values, d_period)
        last_kline = self.history[-1]
        start_time = int(last_kline.data[0].start)
        return StochRSISchema(
            value=[round(k, 2), round(d, 2)],
            kline_ms=start_time
        )

    def predict_rsi(
            self,
            side: str,
            rsi: float,
            target_range: str,
            period: int = 14,
    ) -> Union[PredictRSIResultSchema, None]:

        """
       Подбор значения close, при котором RSI попадет в указанный диапазон.
       direction: "buy" — ищем понижение close, "sell" — повышение
       """

        history = self.history[-period:]
        closes = [float(i.data[0].close) for i in history]
        start_time = int(history[-1].data[0].start)

        from_lev, to_lev = map(float, target_range.split('-'))
        original_close = closes[-1]

        step = abs(original_close) * 0.001  # шаг 0.1% от текущей цены
        max_iter = 2000
        sign = -1 if side == "buy" else 1

        for i in range(max_iter):
            test_close = original_close + sign * step * i
            test_closes = closes[:-1] + [test_close]

            test_rsi = self._calculate_RSI(
                test_closes,
                start_time,
                period
            ).value

            if from_lev <= test_rsi <= to_lev:
                delta_percent = ((test_close - original_close) / original_close) * 100
                return PredictRSIResultSchema(
                    side=side,
                    rate=round(test_close, 2),
                    percent=round(delta_percent, 2),
                    kline_ms=start_time,
                    interval=self.interval,
                    rsi=rsi
                )

    def _calculate_stoch_RSI_from_history(
            self,
            history,
            rsi_period: int,
            stoch_period: int,
            k_period: int,
            d_period: int
    ) -> StochRSISchema:
        """
        Расчёт Stochastic RSI по переданной истории свечей (без использования self.history)
        """

        # Вычисляем RSI серию
        rsi_series = []
        for i in range(len(history) - rsi_period + 1):
            sub_history = history[i:i + rsi_period]
            closes = [float(kline.data[0].close) for kline in sub_history]
            start_time = int(sub_history[-1].data[0].start)
            rsi = self._calculate_RSI(closes, start_time, rsi_period)
            rsi_series.append(rsi.value)

        if len(rsi_series) < stoch_period + k_period + d_period - 1:
            raise ValueError("Недостаточно RSI значений")

        # Stoch RSI
        stoch_rsi_series = []
        for i in range(len(rsi_series) - stoch_period + 1):
            window = rsi_series[i:i + stoch_period]
            rsi_now = window[-1]
            rsi_min = min(window)
            rsi_max = max(window)
            if rsi_max == rsi_min:
                stoch = 0.0
            else:
                stoch = (rsi_now - rsi_min) / (rsi_max - rsi_min)
            stoch_rsi_series.append(stoch * 100)

        # K
        k = simple_sma(stoch_rsi_series, k_period)

        # D
        d_values = [
            simple_sma(stoch_rsi_series[i - k_period + 1:i + 1], k_period)
            for i in range(k_period - 1, len(stoch_rsi_series))
        ]
        d = simple_sma(d_values, d_period)

        last_kline = history[-1]
        start_time = int(last_kline.data[0].start)

        return StochRSISchema(
            value=[round(k, 2), round(d, 2)],
            kline_ms=start_time
        )

    def predict_stoch_rsi(
            self,
            side: str,
            target_range: str,
            rsi_period: int = 14,
            stoch_period: int = 14,
            k_period: int = 3,
            d_period: int = 3,
            max_iter: int = 2000
    ) -> Union[PredictStochRSIResultSchema, None]:
        """
        Подбор значения close, при котором %K из Stochastic RSI попадёт в указанный диапазон.
        side: "buy" — снижение цены, "sell" — повышение.
        История не модифицируется.
        """

        from copy import deepcopy

        # Минимально необходимое количество свечей
        min_history = rsi_period + stoch_period + k_period + d_period
        if len(self.history) < min_history:
            raise ValueError("Недостаточно истории для предсказания Stochastic RSI")

        # Копируем историю, чтобы не портить оригинал
        test_history = deepcopy(self.history)
        closes = [float(i.data[0].close) for i in test_history[-rsi_period:]]
        original_close = closes[-1]
        sign = -1 if side == "buy" else 1
        step = abs(original_close) * 0.001  # шаг 0.1%

        from_lev, to_lev = map(float, target_range.split('-'))

        for i in range(max_iter):
            test_close = original_close + sign * step * i

            # Подменяем close только в копии последней свечи
            test_history[-1].data[0].close = test_close

            # Вызываем расчет по временной истории
            stoch_rsi = self._calculate_stoch_RSI_from_history(
                test_history,
                rsi_period=rsi_period,
                stoch_period=stoch_period,
                k_period=k_period,
                d_period=d_period
            )


            k_value = stoch_rsi.value[0]
            d_value = stoch_rsi.value[1]
            if from_lev <= k_value <= to_lev:
                delta_percent = ((test_close - original_close) / original_close) * 100
                return PredictStochRSIResultSchema(
                    side=side,
                    rate=round(test_close, 2),
                    percent=round(delta_percent, 2),
                    kline_ms=stoch_rsi.kline_ms,
                    interval=self.interval,
                    k=round(k_value, 2),
                    d=round(d_value, 2)
                )

        return None