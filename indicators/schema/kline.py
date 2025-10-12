from pydantic import BaseModel
from typing import List, Optional

from indicators.utils import ms_to_dt


class CandleSchema(BaseModel):
    dt: Optional[str] = None
    start: int
    end: Optional[int] = None
    interval: int
    open: str
    close: str
    high: str
    low: str
    volume: str
    turnover: str
    confirm: bool = True
    timestamp: Optional[int] = None

    @property
    def start_str(self) -> str:
        """
        Возвращает start в формате: timestamp | дата-время
        """
        dt = ms_to_dt(self.start)
        return f"{self.start} | {dt} UTC"

    @property
    def end_str(self) -> str:
        """
        Возвращает start в формате: timestamp | дата-время
        """
        dt = ms_to_dt(self.end)
        return f"{self.end} | {dt} UTC"

class KlineSchema(BaseModel):
    topic: str
    symbol: str
    interval: int
    data: List[CandleSchema]
    ts: Optional[int] = None
    type: Optional[str] = None