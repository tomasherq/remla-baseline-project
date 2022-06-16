from time import time
import os
RUNTIMES_LOCATION = "monitoring/metrics/runtimes/runtimes.txt"
MONITOR_FILE = "monitoring/resources/monitor.txt"


def register_timestamp(job_executed, stage="start"):
    with open(RUNTIMES_LOCATION, "a") as file_write:
        file_write.write(f'{job_executed.split("src/")[1].split(".py")[0]};{time()};{stage}\n')


def start_execution(job_executed):
    if os.path.exists(RUNTIMES_LOCATION):
        creation_time = int(os.path.getctime(RUNTIMES_LOCATION)*10e5)
        os.rename(RUNTIMES_LOCATION, RUNTIMES_LOCATION.replace(".txt", f"{creation_time}.txt"))

    register_timestamp(job_executed)
    with open(MONITOR_FILE, 'w') as file:
        file.write("")
    os.system("python3 monitoring/monitor_basic_metrics.py &")


def end_execution():
    os.remove(MONITOR_FILE)
