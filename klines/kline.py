from .base import _KlinesBase
from indicators.indicators.ADX import ADXIndicator
from indicators.indicators.MACD import MACDIndicator
from indicators.indicators.RSI import RSIIndicator


class Klines(
    RSIIndicator,
    MACDIndicator,
    ADXIndicator,

    _KlinesBase,
):
    pass