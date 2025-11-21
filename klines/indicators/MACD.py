from typing import List, Optional
import pandas as pd
from klines.base import _KlinesBase
from klines.schema.MACD import MACDSchema


class MACDIndicator(_KlinesBase):
    """
    Индикатор MACD, рассчитанный единообразно:
    - EMA с adjust=False
    - MACD/Signal/Hist считаются по всей истории, а не по "скользящим окнам"
    """

    # Параметры MACD по умолчанию
    FAST_SPAN = 12
    SLOW_SPAN = 26
    SIGNAL_SPAN = 9

    @classmethod
    def _build_macd_df(
        cls,
        closes: List[float],
        span_fast: int = FAST_SPAN,
        span_slow: int = SLOW_SPAN,
        span_signal: int = SIGNAL_SPAN,
    ) -> pd.DataFrame:
        """
        Строит DataFrame с колонками:
        close, EMA_fast, EMA_slow, MACD, Signal, Hist
        На ВСЮ историю, как это делает нормальный терминал.
        """
        df = pd.DataFrame(closes, columns=["close"])

        # EMA
        df["EMA_fast"] = df["close"].ewm(span=span_fast, adjust=False).mean()
        df["EMA_slow"] = df["close"].ewm(span=span_slow, adjust=False).mean()

        # MACD и сигнальная
        df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
        df["Signal"] = df["MACD"].ewm(span=span_signal, adjust=False).mean()
        df["Hist"] = df["MACD"] - df["Signal"]

        return df

    @classmethod
    def _last_macd_schema(
        cls,
        df: pd.DataFrame,
        kline_ms: int,
        round_digits: int = 2,
    ) -> MACDSchema:
        """
        Берёт последний ряд из df и упаковывает в MACDSchema.
        """
        last = df.iloc[-1]
        return MACDSchema(
            hist=round(float(last["Hist"]), round_digits),
            macd=round(float(last["MACD"]), round_digits),
            sign=round(float(last["Signal"]), round_digits),
            kline_ms=kline_ms,
        )

    # -------------------------------------------------------------------------
    # Текущее значение MACD
    # -------------------------------------------------------------------------
    def current_MACD(
        self,
        min_bars: int = 50,
    ) -> Optional[MACDSchema]:
        """
        Текущее значение MACD по всей доступной истории.
        min_bars — минимальное количество свечей для более-менее стабильного MACD.
        """
        if len(self.history) < min_bars:
            return None  # или можешь бросать исключение, если так удобнее

        closes = [float(i.data[0].close) for i in self.history]
        df = self._build_macd_df(closes)

        kline_ms = int(self.history[-1].data[0].start)
        return self._last_macd_schema(df, kline_ms)

    # -------------------------------------------------------------------------
    # История MACD
    # -------------------------------------------------------------------------
    def history_MACD(
        self,
        length: int = 30,
        min_bars: int = 50,
    ) -> List[MACDSchema]:
        """
        Возвращает последние `length` значений MACD.
        MACD считается по всей истории, а не по скользящим окнам.
        """
        macd_values: List[MACDSchema] = []

        total_candles = len(self.history)
        if total_candles < max(min_bars, length):
            return macd_values

        closes = [float(i.data[0].close) for i in self.history]
        df = self._build_macd_df(closes)

        # Берём последние `length` записей
        length = min(length, len(df))
        start_idx = len(df) - length

        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            candle = self.history[idx]  # тот же индекс для времени
            kline_ms = int(candle.data[0].start)

            macd_values.append(
                MACDSchema(
                    macd=round(float(row["MACD"]), 2),
                    sign=round(float(row["Signal"]), 2),
                    hist=round(float(row["Hist"]), 2),
                    kline_ms=kline_ms,
                )
            )

        return macd_values

    # -------------------------------------------------------------------------
    # Прогноз MACD (наивная линейная экстраполяция цены)
    # -------------------------------------------------------------------------
    def predict_MACD(
        self,
        steps_ahead: int = 7,
        trend_window: int = 5,
    ) -> List[MACDSchema]:
        """
        Теоретический прогноз MACD на N свечей вперёд.
        Использует линейную экстраполяцию цены по последним трендам.
        Очень грубая модель, использовать с осторожностью.
        """
        closes = [float(i.data[0].close) for i in self.history]
        if len(closes) < 30:
            raise ValueError("Недостаточно данных для прогноза (нужно хотя бы 30 свечей)")

        df = self._build_macd_df(closes)

        # Текущие значения
        last_close = float(df["close"].iloc[-1])
        ema_fast = float(df["EMA_fast"].iloc[-1])
        ema_slow = float(df["EMA_slow"].iloc[-1])
        signal = float(df["Signal"].iloc[-1])

        # Коэффициенты сглаживания
        alpha_fast = 2 / (self.FAST_SPAN + 1)
        alpha_slow = 2 / (self.SLOW_SPAN + 1)
        alpha_signal = 2 / (self.SIGNAL_SPAN + 1)

        # Линейный тренд
        trend = (df["close"].iloc[-1] - df["close"].iloc[-trend_window]) / trend_window

        macd_values: List[MACDSchema] = []
        start_time = int(self.history[-1].data[0].start)
        step_ms = self.interval * 60_000  # минутный интервал в мс

        for step in range(1, steps_ahead + 1):
            next_close = last_close + trend * step

            # обновляем EMA
            ema_fast = (next_close - ema_fast) * alpha_fast + ema_fast
            ema_slow = (next_close - ema_slow) * alpha_slow + ema_slow

            macd = ema_fast - ema_slow
            signal = (macd - signal) * alpha_signal + signal
            hist = macd - signal

            macd_values.append(
                MACDSchema(
                    macd=round(float(macd), 4),
                    sign=round(float(signal), 4),
                    hist=round(float(hist), 4),
                    kline_ms=start_time + step * step_ms,
                )
            )

        return macd_values

    # -------------------------------------------------------------------------
    # Поиск ближайшего разворота MACD (пересечение с Signal)
    # -------------------------------------------------------------------------
    def predict_MACD_reversal(
        self,
        trend_window: int = 5,
        max_steps: int = 50,
        scenario: str = "auto",
    ):
        """
        Поиск вероятного разворота MACD (пересечение MACD и сигнальной линии).
        scenario: "auto", "bullish", "bearish"
        auto: направление выбирается на основе знака текущей гистограммы.
        """
        closes = [float(i.data[0].close) for i in self.history]
        if len(closes) < 30:
            raise ValueError("Недостаточно данных для анализа разворота")

        df = self._build_macd_df(closes)

        # Текущие значения
        last_close = float(df["close"].iloc[-1])
        ema_fast = float(df["EMA_fast"].iloc[-1])
        ema_slow = float(df["EMA_slow"].iloc[-1])
        signal = float(df["Signal"].iloc[-1])
        hist = float(df["Hist"].iloc[-1])

        # Коэффициенты сглаживания
        alpha_fast = 2 / (self.FAST_SPAN + 1)
        alpha_slow = 2 / (self.SLOW_SPAN + 1)
        alpha_signal = 2 / (self.SIGNAL_SPAN + 1)

        # Направление тренда по сценарию
        base_trend = (df["close"].iloc[-1] - df["close"].iloc[-trend_window]) / trend_window

        if scenario == "bullish":
            trend = abs(base_trend) if base_trend != 0 else abs(df["close"].pct_change().iloc[-1]) * last_close
        elif scenario == "bearish":
            trend = -abs(base_trend) if base_trend != 0 else -abs(df["close"].pct_change().iloc[-1]) * last_close
        else:  # auto
            if hist > 0:
                trend = -abs(base_trend) if base_trend != 0 else -abs(df["close"].pct_change().iloc[-1]) * last_close
            else:
                trend = abs(base_trend) if base_trend != 0 else abs(df["close"].pct_change().iloc[-1]) * last_close

        start_time = int(self.history[-1].data[0].start)
        step_ms = self.interval * 60_000

        prev_hist = hist

        for step in range(1, max_steps + 1):
            next_close = last_close + trend * step

            ema_fast = (next_close - ema_fast) * alpha_fast + ema_fast
            ema_slow = (next_close - ema_slow) * alpha_slow + ema_slow

            macd = ema_fast - ema_slow
            signal = (macd - signal) * alpha_signal + signal
            hist = macd - signal

            # смена знака гистограммы = пересечение MACD и Signal
            if (prev_hist > 0 and hist <= 0) or (prev_hist < 0 and hist >= 0):
                return {
                    "step": step,
                    "predicted_close": round(float(next_close), 4),
                    "macd": round(float(macd), 4),
                    "signal": round(float(signal), 4),
                    "hist": round(float(hist), 4),
                    "time_ms": start_time + step * step_ms,
                    "direction": "bearish" if prev_hist > 0 else "bullish",
                }

            prev_hist = hist

        return None  # разворот не найден в пределах max_steps
