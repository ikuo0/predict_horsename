
from datetime import datetime, timedelta

def now() -> datetime:
    return datetime.now() # .strftime("%Y-%m-%d %H:%M:%S")

def elapsed(start_time: datetime) -> tuple[datetime, timedelta]:
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    return end_time, elapsed_time

def to_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

class TimeMeasurer:
    def __init__(self):
        self.start_time = now()

    def finish(self) -> tuple[datetime, timedelta]:
        return elapsed(self.start_time)
