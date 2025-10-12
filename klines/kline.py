from klines.base import _KlinesBase
from klines.indicators.ADX import ADXIndicator
from klines.indicators.MACD import MACDIndicator
from klines.indicators.RSI import RSIIndicator


class Klines(
    RSIIndicator,
    MACDIndicator,
    ADXIndicator,

    _KlinesBase,
):
    pass