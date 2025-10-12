from .base import _KlinesBase
from .indicators.ADX import ADXIndicator
from .indicators.MACD import MACDIndicator
from .indicators.RSI import RSIIndicator


class Klines(
    RSIIndicator,
    MACDIndicator,
    ADXIndicator,

    _KlinesBase,
):
    pass