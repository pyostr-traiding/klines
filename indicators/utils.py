import datetime


def ms_to_dt(ms):
    ms =  ms / 1000
    dt = datetime.datetime.fromtimestamp(ms, datetime.UTC)
    return dt.strftime('%Y-%m-%d %H:%M:%S')
