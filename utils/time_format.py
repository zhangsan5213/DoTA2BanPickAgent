from datetime import datetime

def format_timestamp(ts):
    """
    将时间戳转换为 YYYY-MM-DD-HH-mm-ss 格式
    """
    # 将时间戳转换为 datetime 对象
    dt_object = datetime.fromtimestamp(ts)
    
    # 格式化为指定的字符串
    # %Y: 年, %m: 月, %d: 日, %H: 时, %M: 分, %S: 秒
    return dt_object.strftime('%Y-%m-%d %H:%M:%S')