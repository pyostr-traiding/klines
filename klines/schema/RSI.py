from typing import List

from klines.schema.base_indicator import BaseIndicatorSchema


class RSISchema(BaseIndicatorSchema):
    """
    Схема RSI
    """
    value: float
    interval: int


class StochRSISchema(BaseIndicatorSchema):
    """
    Схема Stoch RSI
    """
    value: List[float]


class PredictRSIResultSchema(BaseIndicatorSchema):
    """
    Предсказание значения
    """
    side: str
    rate: float
    percent: float
    interval: int
    rsi: float


class PredictStochRSIResultSchema(BaseIndicatorSchema):
    """
    Предсказание значения
    """
    side: str
    rate: float
    percent: float
    interval: int
    k: float
    d: float
