import psutil
from prometheus_client import Gauge

process = psutil.Process()

# System/process metrics (semplici ma utili) - custom prefix to avoid Prometheus collisions
PROCESS_RSS_BYTES = Gauge("mlops_process_resident_memory_bytes", "Resident memory size in bytes")
PROCESS_CPU_PERCENT = Gauge("mlops_process_cpu_percent", "Process CPU percent since last call")
PROCESS_NUM_THREADS = Gauge("mlops_process_num_threads", "Number of threads in the process")
PROCESS_OPEN_FDS = Gauge("mlops_process_open_fds", "Number of open file descriptors (0 if not supported)")

_last_cpu_call = False


def update_system_metrics() -> None:
    global _last_cpu_call

    PROCESS_RSS_BYTES.set(process.memory_info().rss)
    PROCESS_NUM_THREADS.set(process.num_threads())

    # open_fds non esiste su Windows; su Linux ok. Su Cloud Run (Linux) ok.
    try:
        PROCESS_OPEN_FDS.set(process.num_fds())
    except Exception:
        PROCESS_OPEN_FDS.set(0)

    # cpu_percent: la prima chiamata spesso ritorna 0.0 (baseline)
    if not _last_cpu_call:
        process.cpu_percent(interval=None)
        _last_cpu_call = True
    else:
        PROCESS_CPU_PERCENT.set(process.cpu_percent(interval=None))
