from typing import List, Optional
from klines.base import _KlinesBase
from klines.schema.MACD import MACDSchema


class MACDIndicator(_KlinesBase):
    FAST_SPAN = 12
    SLOW_SPAN = 26
    SIGNAL_SPAN = 9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # внутренние EMA (ленивая инициализация)
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.signal: Optional[float] = None

        # история MACD
        self.macd_history: List[MACDSchema] = []

        # коэффициенты сглаживания
        self.alpha_fast = 2 / (self.FAST_SPAN + 1)
        self.alpha_slow = 2 / (self.SLOW_SPAN + 1)
        self.alpha_signal = 2 / (self.SIGNAL_SPAN + 1)

    # -------------------------------------------------------------------------
    # ИНИЦИАЛИЗАЦИЯ ИЗ HISTORY (lenient)
    # -------------------------------------------------------------------------
    def _lazy_init_from_history(self):
        """
        Выполняется один раз: инициализация EMA по реальной истории,
        расчёт всего MACD, заполнение macd_history.
        """

        if self.ema_fast is not None:
            return  # уже инициализировано

        if not self.history:
            return

        closes = [float(i.data[0].close) for i in self.history]
        n = len(closes)
from typing import List, Optional
from klines.base import _KlinesBase
from klines.schema.MACD import MACDSchema


class MACDIndicator(_KlinesBase):
    FAST_SPAN = 12
    SLOW_SPAN = 26
    SIGNAL_SPAN = 9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # внутренние EMA (ленивая инициализация)
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.signal: Optional[float] = None

        # история MACD
        self.macd_history: List[MACDSchema] = []

        # коэффициенты сглаживания
        self.alpha_fast = 2 / (self.FAST_SPAN + 1)
        self.alpha_slow = 2 / (self.SLOW_SPAN + 1)
        self.alpha_signal = 2 / (self.SIGNAL_SPAN + 1)

    # -------------------------------------------------------------------------
    # ИНИЦИАЛИЗАЦИЯ ИЗ HISTORY (lenient)
    # -------------------------------------------------------------------------
    def _lazy_init_from_history(self):
        """
        Выполняется один раз: инициализация EMA по реальной истории,
        расчёт всего MACD, заполнение macd_history.
        """

        if self.ema_fast is not None:
            return  # уже инициализировано

        if not self.history:
            return

        closes = [float(i.data[0].close) for i in self.history]
        n = len(closes)

        # начальные значения EMA по первым данным
        self.ema_fast = closes[0]
        self.ema_slow = closes[0]
        self.signal = 0.0

        self.macd_history.clear()

        for idx in range(n):
            close = closes[idx]
            kline_ms = int(self.history[idx].data[0].start)

            # EMA fast/slow
            self.ema_fast = (close - self.ema_fast) * self.alpha_fast + self.ema_fast
            self.ema_slow = (close - self.ema_slow) * self.alpha_slow + self.ema_slow

            macd = self.ema_fast - self.ema_slow
            self.signal = (macd - self.signal) * self.alpha_signal + self.signal

            hist = macd - self.signal

            self.macd_history.append(
                MACDSchema(
                    macd=round(float(macd), 4),
                    sign=round(float(self.signal), 4),
                    hist=round(float(hist), 4),
                    kline_ms=kline_ms,
                )
            )

    # -------------------------------------------------------------------------
    # ТЕКУЩЕЕ ЗНАЧЕНИЕ MACD (часто вызывается)
    # -------------------------------------------------------------------------
    def current_MACD(self) -> Optional[MACDSchema]:
        """
        Гарантированно работает без предварительного update().
        Инициализирует MACD по history при первом вызове.
        """
        if self.ema_fast is None:
            self._lazy_init_from_history()

        if not self.macd_history:
            return None

        return self.macd_history[-1]

    # -------------------------------------------------------------------------
    # ИСТОРИЯ MACD
    # -------------------------------------------------------------------------
    def history_MACD(self, length: int = 30) -> List[MACDSchema]:
        if self.ema_fast is None:
            self._lazy_init_from_history()

        return self.macd_history[-length:]

    # -------------------------------------------------------------------------
    # ПРОГНОЗ MACD
    # -------------------------------------------------------------------------
    def predict_MACD(
        self,
        steps_ahead: int = 7,
        trend_window: int = 5,
    ) -> List[MACDSchema]:

        if self.ema_fast is None:
            self._lazy_init_from_history()

        if not self.history or len(self.history) < trend_window + 2:
            return []

        last_close = float(self.history[-1].data[0].close)
        prev_close = float(self.history[-trend_window].data[0].close)

        trend = (last_close - prev_close) / trend_window

        # копии EMA, чтобы не портить реальные
        ema_fast = self.ema_fast
        ema_slow = self.ema_slow
        signal = self.signal

        results = []
        step_ms = self.interval * 60_000
        start_ms = int(self.history[-1].data[0].start)

        for step in range(1, steps_ahead + 1):
            next_close = last_close + trend * step

            ema_fast = (next_close - ema_fast) * self.alpha_fast + ema_fast
            ema_slow = (next_close - ema_slow) * self.alpha_slow + ema_slow

            macd = ema_fast - ema_slow
            signal = (macd - signal) * self.alpha_signal + signal

            hist = macd - signal

            results.append(
                MACDSchema(
                    macd=round(float(macd), 4),
                    sign=round(float(signal), 4),
                    hist=round(float(hist), 4),
                    kline_ms=start_ms + step * step_ms,
                )
            )

        return results

    # -------------------------------------------------------------------------
    # ПОИСК РАЗВОРОТА MACD
    # -------------------------------------------------------------------------
    def predict_MACD_reversal(
        self,
        trend_window: int = 5,
        max_steps: int = 50,
        scenario: str = "auto",
    ):
        if self.ema_fast is None:
            self._lazy_init_from_history()

        if not self.history or len(self.history) < trend_window + 2:
            return None

        last_close = float(self.history[-1].data[0].close)
        prev_close = float(self.history[-trend_window].data[0].close)

        base_trend = (last_close - prev_close) / trend_window

        current_hist = self.macd_history[-1].hist if self.macd_history else 0

        # определяем направление
        if scenario == "bullish":
            trend = abs(base_trend)
        elif scenario == "bearish":
            trend = -abs(base_trend)
        else:
            trend = -abs(base_trend) if current_hist > 0 else abs(base_trend)

        ema_fast = self.ema_fast
        ema_slow = self.ema_slow
        signal = self.signal
        prev_hist = current_hist

        step_ms = self.interval * 60_000
        start_ms = int(self.history[-1].data[0].start)

        for step in range(1, max_steps + 1):
            next_close = last_close + trend * step

            ema_fast = (next_close - ema_fast) * self.alpha_fast + ema_fast
            ema_slow = (next_close - ema_slow) * self.alpha_slow + ema_slow

            macd = ema_fast - ema_slow
            signal = (macd - signal) * self.alpha_signal + signal

            hist = macd - signal

            # пересечение
            if (prev_hist > 0 and hist <= 0) or (prev_hist < 0 and hist >= 0):
                return {
                    "step": step,
                    "predicted_close": round(float(next_close), 4),
                    "macd": round(float(macd), 4),
                    "signal": round(float(signal), 4),
                    "hist": round(float(hist), 4),
                    "time_ms": start_ms + step * step_ms,
                    "direction": "bearish" if prev_hist > 0 else "bullish",
                }

            prev_hist = hist

        return None

        # начальные значения EMA по первым данным
        self.ema_fast = closes[0]
        self.ema_slow = closes[0]
        self.signal = 0.0

        self.macd_history.clear()

        for idx in range(n):
            close = closes[idx]
            kline_ms = int(self.history[idx].data[0].start)

            # EMA fast/slow
            self.ema_fast = (close - self.ema_fast) * self.alpha_fast + self.ema_fast
            self.ema_slow = (close - self.ema_slow) * self.alpha_slow + self.ema_slow

            macd = self.ema_fast - self.ema_slow
            self.signal = (macd - self.signal) * self.alpha_signal + self.signal

            hist = macd - self.signal

            self.macd_history.append(
                MACDSchema(
                    macd=round(float(macd), 4),
                    sign=round(float(self.signal), 4),
                    hist=round(float(hist), 4),
                    kline_ms=kline_ms,
                )
            )

    # -------------------------------------------------------------------------
    # ТЕКУЩЕЕ ЗНАЧЕНИЕ MACD (часто вызывается)
    # -------------------------------------------------------------------------
    def current_MACD(self) -> Optional[MACDSchema]:
        """
        Гарантированно работает без предварительного update().
        Инициализирует MACD по history при первом вызове.
        """
        if self.ema_fast is None:
            self._lazy_init_from_history()

        if not self.macd_history:
            return None

        return self.macd_history[-1]

    # -------------------------------------------------------------------------
    # ИСТОРИЯ MACD
    # -------------------------------------------------------------------------
    def history_MACD(self, length: int = 30) -> List[MACDSchema]:
        if self.ema_fast is None:
            self._lazy_init_from_history()

        return self.macd_history[-length:]

    # -------------------------------------------------------------------------
    # ПРОГНОЗ MACD
    # -------------------------------------------------------------------------
    def predict_MACD(
        self,
        steps_ahead: int = 7,
        trend_window: int = 5,
    ) -> List[MACDSchema]:

        if self.ema_fast is None:
            self._lazy_init_from_history()

        if not self.history or len(self.history) < trend_window + 2:
            return []

        last_close = float(self.history[-1].data[0].close)
        prev_close = float(self.history[-trend_window].data[0].close)

        trend = (last_close - prev_close) / trend_window

        # копии EMA, чтобы не портить реальные
        ema_fast = self.ema_fast
        ema_slow = self.ema_slow
        signal = self.signal

        results = []
        step_ms = self.interval * 60_000
        start_ms = int(self.history[-1].data[0].start)

        for step in range(1, steps_ahead + 1):
            next_close = last_close + trend * step

            ema_fast = (next_close - ema_fast) * self.alpha_fast + ema_fast
            ema_slow = (next_close - ema_slow) * self.alpha_slow + ema_slow

            macd = ema_fast - ema_slow
            signal = (macd - signal) * self.alpha_signal + signal

            hist = macd - signal

            results.append(
                MACDSchema(
                    macd=round(float(macd), 4),
                    sign=round(float(signal), 4),
                    hist=round(float(hist), 4),
                    kline_ms=start_ms + step * step_ms,
                )
            )

        return results

    # -------------------------------------------------------------------------
    # ПОИСК РАЗВОРОТА MACD
    # -------------------------------------------------------------------------
    def predict_MACD_reversal(
        self,
        trend_window: int = 5,
        max_steps: int = 50,
        scenario: str = "auto",
    ):
        if self.ema_fast is None:
            self._lazy_init_from_history()

        if not self.history or len(self.history) < trend_window + 2:
            return None

        last_close = float(self.history[-1].data[0].close)
        prev_close = float(self.history[-trend_window].data[0].close)

        base_trend = (last_close - prev_close) / trend_window

        current_hist = self.macd_history[-1].hist if self.macd_history else 0

        # определяем направление
        if scenario == "bullish":
            trend = abs(base_trend)
        elif scenario == "bearish":
            trend = -abs(base_trend)
        else:
            trend = -abs(base_trend) if current_hist > 0 else abs(base_trend)

        ema_fast = self.ema_fast
        ema_slow = self.ema_slow
        signal = self.signal
        prev_hist = current_hist

        step_ms = self.interval * 60_000
        start_ms = int(self.history[-1].data[0].start)

        for step in range(1, max_steps + 1):
            next_close = last_close + trend * step

            ema_fast = (next_close - ema_fast) * self.alpha_fast + ema_fast
            ema_slow = (next_close - ema_slow) * self.alpha_slow + ema_slow

            macd = ema_fast - ema_slow
            signal = (macd - signal) * self.alpha_signal + signal

            hist = macd - signal

            # пересечение
            if (prev_hist > 0 and hist <= 0) or (prev_hist < 0 and hist >= 0):
                return {
                    "step": step,
                    "predicted_close": round(float(next_close), 4),
                    "macd": round(float(macd), 4),
                    "signal": round(float(signal), 4),
                    "hist": round(float(hist), 4),
                    "time_ms": start_ms + step * step_ms,
                    "direction": "bearish" if prev_hist > 0 else "bullish",
                }

            prev_hist = hist

        return None
