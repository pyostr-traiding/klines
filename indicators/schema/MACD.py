from indicators.schema.base_indicator import BaseIndicatorSchema


class MACDSchema(BaseIndicatorSchema):
    """
    Схема MACD
    """
    hist: float
    macd: float
    sign: float
