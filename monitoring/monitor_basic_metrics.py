import psutil
import os
from collections import defaultdict
from statistics import mean
import json

RESULTS_DIRECTION = "monitoring/metrics/results_execution.json"
MONITOR_FILE = "monitoring/resources/monitor.txt"


INTERVAL = 1

STARTING_READS = psutil.disk_io_counters()

reads = STARTING_READS.read_count
writes = STARTING_READS.write_count

metrics_used = defaultdict(list)


while True:
    if not os.path.exists(MONITOR_FILE):
        break

    cpu_usage = psutil.cpu_percent(interval=INTERVAL)
    swap_memory = psutil.swap_memory()
    ram_memory = psutil.virtual_memory()
    disk_usage = psutil.disk_usage("/")
    io_metrics = psutil.disk_io_counters()

    # This is the current if we want cummulative we can just do it at the end?
    # I thinks is nice to have data points tho to make maybe some graphs
    reads = io_metrics.read_count-reads
    writes = io_metrics.write_count-writes

    metrics_used["cpu_usage"].append(cpu_usage)
    metrics_used["swap_memory"].append(swap_memory.percent)
    metrics_used["ram_memory"].append(ram_memory.percent)
    metrics_used["disk_usage"].append(disk_usage.percent)
    metrics_used["reads"].append(reads)
    metrics_used["writes"].append(writes)

summary_results = {}

for key, values in metrics_used.items():

    if key == "reads" or key == "writes":
        summary_results[key] = sum(values)
    else:
        summary_results[key] = mean(values)

metrics_used["total"] = summary_results

if os.path.exists(RESULTS_DIRECTION):
    creation_time = int(os.path.getctime(RESULTS_DIRECTION)*10e5)
    os.rename(RESULTS_DIRECTION, RESULTS_DIRECTION.replace(".json", f"-{creation_time}.json"))

with open(RESULTS_DIRECTION, "w") as file_write:
    file_write.write(json.dumps(metrics_used, indent=4))
