# FlexKV Prometheus Metrics Documentation

FlexKV integrates a [Prometheus](https://prometheus.io/)-based runtime metrics monitoring framework, covering critical paths in both the Python and C++ layers. The framework is embedded in the FlexKV runtime in a **zero-intrusion** manner — users simply set the environment variable `FLEXKV_ENABLE_METRICS=1` to automatically collect core metrics such as cache hits, memory pool status, and data transfers during application runtime, exposing them via standard HTTP endpoints for Prometheus scraping and Grafana visualization.

---

## 1. Configuration

### 1.1 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLEXKV_ENABLE_METRICS` | `0` | Enable metrics collection (set to `1` to enable, disabled by default) |
| `FLEXKV_PY_METRICS_PORT` | `8080` | Python metrics HTTP server port |
| `FLEXKV_CPP_METRICS_PORT` | `8081` | C++ metrics HTTP server port |
| `FLEXKV_PY_METRICS_MULTIPROC_DIR` | `flexkv-multiproc-<port>` under the system temp dir | Shared directory for cross-process metrics; rarely needs to be set |

### 1.2 Configuration

```bash
# Enable FlexKV metrics collection
export FLEXKV_ENABLE_METRICS=1

# Custom ports (optional)
export FLEXKV_PY_METRICS_PORT=8080
export FLEXKV_CPP_METRICS_PORT=8081
```

---

## 2. Metrics Reference

### 2.1 Python Runtime Metrics (`flexkv_py_*`)

Python metrics are recorded by `GlobalCacheEngine` in `cache_engine.py` and collected via `FlexKVMetricsCollector`.

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `flexkv_py_cache_hit_blocks_total` | Counter | `device` | Total number of cache-hit blocks |
| `flexkv_py_cache_miss_blocks_total` | Counter | - | Total number of cache-miss blocks (missed at all levels) |
| `flexkv_py_transfer_blocks_total` | Counter | `transfer_type`, `operation` | Total number of transferred blocks |
| `flexkv_py_transfer_ops_total` | Counter | `transfer_type`, `operation` | Number of transfer operations |
| `flexkv_py_transfer_bytes_total` | Counter | `transfer_type`, `operation` | Total bytes transferred |
| `flexkv_py_mempool_total_blocks` | Gauge | `device` | Total blocks in memory pool |
| `flexkv_py_mempool_free_blocks` | Gauge | `device` | Free blocks in memory pool |
| `flexkv_py_evicted_blocks_total` | Counter | `device` | Total number of evicted blocks |
| `flexkv_py_allocated_blocks_total` | Counter | `device` | Total number of allocated blocks |
| `flexkv_py_allocation_failures_total` | Counter | `mode` | Number of allocation failures |
| `flexkv_py_transfer_wait_duration_seconds` | Histogram | `transfer_type` | Time a transfer spent queued before launch, in seconds |
| `flexkv_py_transfer_xfer_duration_seconds` | Histogram | `transfer_type` | Time spent moving the data, in seconds |
| `flexkv_py_transfer_e2e_duration_seconds` | Histogram | `transfer_type` | Time from submit to completion, in seconds |
| `flexkv_py_metrics_server_info` | Gauge | `pid` | Always 1; identifies which process serves the Python endpoint |

**Notes on the duration metrics**

The three histograms are recorded by the transfer side (`TransferEngine`), the only place that sees the full timing. They therefore carry **only `transfer_type`, not `operation`** - the get/put split lives on the `flexkv_py_transfer_*_total` counters above, which the engine side records per task type. The two families cannot be joined on `operation`.

The three durations nest: `wait` is queueing, `xfer` is the actual data movement, `e2e` is submit to completion and includes both. Buckets span 0.5ms to 30s, covering an H2D hit through a cold SSD read.

**Multi-process aggregation**

FlexKV records from more than one process: `GlobalCacheEngine` lives in the engine process, while `TransferEngine` - and therefore all transfer timing - runs in the spawned TransferManager subprocess. With `FLEXKV_ENABLE_METRICS=1`, FlexKV sets `PROMETHEUS_MULTIPROC_DIR` automatically; every process writes into that shared directory and the process that binds `FLEXKV_PY_METRICS_PORT` exposes the merged result.

Consequences:

- Multiple FlexKV processes on one node (TP / PP / multiple workers) **share a single port**. Whoever binds first serves, and the values are the merge of all of them.
- Only the main process starts the HTTP server; subprocesses just write to the shared directory.
- If the port is taken by a **non-FlexKV service**, startup raises with a message to change the port instead of silently dropping metrics.
- `.db` files left behind by exited processes are purged on the next start, so dead processes stop contributing values.

---

### 2.2 C++ Runtime Metrics (`flexkv_cpp_*`)

C++ metrics are managed by the `MetricsManager` singleton, primarily instrumented in RadixTree cache operations and data transfers.

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `flexkv_cpp_cache_ops_total` | Counter | `operation` | RadixTree cache operation count |
| `flexkv_cpp_cache_blocks_total` | Counter | `operation` | Blocks involved in RadixTree cache operations |

---

## 3. Monitoring Stack Deployment

### 3.1 Directory Structure

```
FlexKV/monitoring/
├── docker-compose.yml         # Prometheus + Grafana container orchestration
├── prometheus.yml             # Prometheus scrape configuration
└── grafana/
    ├── dashboards/
    │   └── flexkv-demo.json   # Grafana pre-built dashboard
    └── provisioning/
        ├── dashboards/
        │   └── dashboards.yml # Dashboard auto-load configuration
        └── datasources/
            └── prometheus.yml # Datasource auto-configuration
```

### 3.2 Quick Deploy

```bash
# 0. Install Python dependency
pip3 install prometheus_client

# 1. Start FlexKV application with monitoring enabled
export FLEXKV_ENABLE_METRICS=1
python your_flexkv_app.py

# 2. Start Prometheus + Grafana services
cd <path-to-FlexKV>/monitoring
docker compose up -d

# 3. Stop Prometheus + Grafana services
cd <path-to-FlexKV>/monitoring
docker compose stop

# 4. Fully clean up Prometheus + Grafana services
cd <path-to-FlexKV>/monitoring
docker compose down -v
```

### 3.3 Service Access

| Service | URL | Description |
|---|---|---|
| Python Metrics | `http://localhost:8080/metrics` | Python runtime metrics endpoint |
| C++ Metrics | `http://localhost:8081/metrics` | C++ runtime metrics endpoint |
| Prometheus | `http://localhost:9090` | Metrics query interface |
| Grafana | `http://localhost:3000` | Visualization dashboards |

**Quick endpoint verification:**

```bash
# Verify Python metrics endpoint
curl -s http://localhost:8080/metrics | grep flexkv_py_

# Verify C++ metrics endpoint
curl -s http://localhost:8081/metrics | grep flexkv_cpp_
```

### 3.4 Accessing Grafana Dashboards

1. Open your browser and navigate to `http://localhost:3000`
2. Log in with default credentials: username `admin`, password `admin`
3. Go to **Dashboards → FlexKV Demo** to view the pre-built dashboard

**Pre-built dashboard panels:**

| Section | Panel | Description |
|---|---|---|
| Python Runtime Metrics | Cache Hit/Miss Rate | Cache hit/miss rate |
| Python Runtime Metrics | Memory Pool Blocks | Memory pool block statistics |
| Python Runtime Metrics | Transfer Throughput | Data transfer throughput |
| C++ Runtime Metrics | Cache Operations Rate | Cache operation rate |
| C++ Runtime Metrics | Cache Blocks Rate | Cache blocks operation rate |

> Users can create custom panels and configure PromQL queries as needed.
