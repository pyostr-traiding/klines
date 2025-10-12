from pydantic import BaseModel

from indicators.utils import ms_to_dt


class BaseIndicatorSchema(BaseModel):
    """
    Базовая схема индикатора
    """

    kline_ms: int

    @property
    def kline_ms_str(self) -> str:
        """
        Возвращает start в формате: timestamp | дата-время
        """
        dt = ms_to_dt(self.kline_ms)
        return f"{self.kline_ms} | {dt}"