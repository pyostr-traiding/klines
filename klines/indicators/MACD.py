from typing import List
import pandas as pd
from klines.base import _KlinesBase
from klines.schema.MACD import MACDSchema


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
        start_time = int(history[-1].data[0].start)
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
            start_time = int(window[-1].data[0].start)
            macd = self._calculate_MACD(closes, start_time)
            macd_values.append(macd)

        return macd_values

    def predict_MACD(
            self,
            steps_ahead: int = 7,
            trend_window: int = 5
    ) -> List[MACDSchema]:
        """
        Теоретический прогноз MACD на N свечей вперёд.
        Использует линейную экстраполяцию цены по последним трендам.
        """
        closes = [float(i.data[0].close) for i in self.history]
        if len(closes) < 30:
            raise ValueError("Недостаточно данных для прогноза (нужно хотя бы 30 свечей)")

        df = pd.DataFrame(closes, columns=["close"])

        # Параметры MACD
        span_12 = 12
        span_26 = 26
        span_signal = 9

        # Расчёт текущих EMA и MACD
        df["EMA_12"] = df["close"].ewm(span=span_12, adjust=False).mean()
        df["EMA_26"] = df["close"].ewm(span=span_26, adjust=False).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["Signal_Line"] = df["MACD"].ewm(span=span_signal, adjust=False).mean()

        last_close = df["close"].iloc[-1]
        ema12 = df["EMA_12"].iloc[-1]
        ema26 = df["EMA_26"].iloc[-1]
        signal = df["Signal_Line"].iloc[-1]

        # Коэффициенты сглаживания
        alpha_12 = 2 / (span_12 + 1)
        alpha_26 = 2 / (span_26 + 1)
        alpha_signal = 2 / (span_signal + 1)

        # Простейший тренд (линейная экстраполяция)
        trend = (df["close"].iloc[-1] - df["close"].iloc[-trend_window]) / trend_window

        macd_values = []
        start_time = int(self.history[-1].data[0].start)
        step_ms = self.interval * 60_000  # шаг времени в мс

        for i in range(1, steps_ahead + 1):
            next_close = last_close + trend * i

            # обновляем EMA
            ema12 = (next_close - ema12) * alpha_12 + ema12
            ema26 = (next_close - ema26) * alpha_26 + ema26

            macd = ema12 - ema26
            signal = (macd - signal) * alpha_signal + signal
            hist = macd - signal

            macd_values.append(MACDSchema(
                macd=round(macd, 4),
                sign=round(signal, 4),
                hist=round(hist, 4),
                kline_ms=start_time + i * step_ms
            ))

        return macd_values
    
    def predict_MACD_reversal(
            self,
            trend_window: int = 5,
            max_steps: int = 50,
            scenario: str = "auto"
    ):
        """
        Поиск вероятного разворота MACD (точка пересечения MACD и сигнальной линии).
        scenario: "auto", "bullish", "bearish"
        """
        closes = [float(i.data[0].close) for i in self.history]
        if len(closes) < 30:
            raise ValueError("Недостаточно данных для анализа разворота")

        df = pd.DataFrame(closes, columns=["close"])

        # --- Параметры MACD ---
        span_12, span_26, span_signal = 12, 26, 9
        alpha_12 = 2 / (span_12 + 1)
        alpha_26 = 2 / (span_26 + 1)
        alpha_signal = 2 / (span_signal + 1)

        # --- Текущие значения ---
        df["EMA_12"] = df["close"].ewm(span=span_12, adjust=False).mean()
        df["EMA_26"] = df["close"].ewm(span=span_26, adjust=False).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["Signal"] = df["MACD"].ewm(span=span_signal, adjust=False).mean()

        last_close = df["close"].iloc[-1]
        ema12 = df["EMA_12"].iloc[-1]
        ema26 = df["EMA_26"].iloc[-1]
        signal = df["Signal"].iloc[-1]
        hist = ema12 - ema26 - signal

        # --- Направление тренда ---
        trend = (df["close"].iloc[-1] - df["close"].iloc[-trend_window]) / trend_window

        if scenario == "bullish":
            trend = abs(trend)
        elif scenario == "bearish":
            trend = -abs(trend)
        elif scenario == "auto":
            # если текущая гистограмма положительная — ищем вниз, иначе вверх
            trend = -abs(trend) if hist > 0 else abs(trend)

        start_time = int(self.history[-1].data[0].start)
        step_ms = self.interval * 60_000

        prev_hist = hist
        for i in range(1, max_steps + 1):
            next_close = last_close + trend * i

            ema12 = (next_close - ema12) * alpha_12 + ema12
            ema26 = (next_close - ema26) * alpha_26 + ema26

            macd = ema12 - ema26
            signal = (macd - signal) * alpha_signal + signal
            hist = macd - signal

            # Проверяем смену знака гистограммы
            if (prev_hist > 0 and hist <= 0) or (prev_hist < 0 and hist >= 0):
                return {
                    "step": i,
                    "predicted_close": round(next_close, 4),
                    "macd": round(macd, 4),
                    "signal": round(signal, 4),
                    "hist": round(hist, 4),
                    "time_ms": start_time + i * step_ms,
                    "direction": "bearish" if prev_hist > 0 else "bullish"
                }

            prev_hist = hist

        return None  # если разворот не найден в пределах max_steps
