from klines.schema.base_indicator import BaseIndicatorSchema


class ADXSchema(BaseIndicatorSchema):
    """
    Схема MACD
    """
    simple: float
    wilder: float
    ema: float
