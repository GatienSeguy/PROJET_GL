from datetime import datetime

date_str = "2025-10-21-21:30:00"
dt = datetime.strptime(date_str, "%Y-%m-%d-%H:%M:%S")

seconds = int(dt.timestamp())
print(seconds)