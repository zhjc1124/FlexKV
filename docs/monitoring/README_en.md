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
| `FLEXKV_PY_METRICS_MULTIPROC_DIR` | derived from port | Python metrics aggregation directory, see [section 3](#3-multi-process-aggregation-python-metrics) |

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
| `flexkv_py_metrics_server_info` | Gauge | - | Always `1`; marks the port as owned by a FlexKV metrics endpoint |

> Except for `flexkv_py_metrics_server_info`, every `flexkv_py_*` metric above is the
> **sum over all FlexKV processes on the node**, not the value of a single process.
> See [section 3](#3-multi-process-aggregation-python-metrics).

---

### 2.2 C++ Runtime Metrics (`flexkv_cpp_*`)

C++ metrics are managed by the `MetricsManager` singleton, primarily instrumented in RadixTree cache operations and data transfers.

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `flexkv_cpp_cache_ops_total` | Counter | `operation` | RadixTree cache operation count |
| `flexkv_cpp_cache_blocks_total` | Counter | `operation` | Blocks involved in RadixTree cache operations |

---

## 3. Multi-Process Aggregation (Python Metrics)

> Deeper reading: [design notes (zh)](./multiprocess_metrics_design_zh.md) ·
> [fix notes (zh)](./multiprocess_metrics_fix_zh.md)

### 3.1 Why it is needed

Python metrics are recorded by `GlobalCacheEngine`, which initializes the collector
in its constructor, and there is one `GlobalCacheEngine` **per engine process**. A
TP/DP deployment, or a FlexKV server with multiple clients, runs several processes
on the same node; each holds its own counters and each tries to bind the same
`FLEXKV_PY_METRICS_PORT`.

Only whichever process wins the bind can expose anything. Before aggregation, what
you saw was not a node-level number but **one arbitrary process** — which one depends
on who binds first and can change on every restart.

### 3.2 How it works

- Every process writes into one shared mmap directory (prometheus_client's
  multiprocess mode). The default path is derived from the port:
  `<tmp>/flexkv-multiproc-<FLEXKV_PY_METRICS_PORT>`, so all processes of one
  deployment share it.
- **Only the main process** starts the HTTP endpoint. Other processes write into the
  directory and are merged into that endpoint.
- If another FlexKV process already owns the port, this process does not bind again
  and treats the endpoint as shared.
- The directory is emptied when a deployment starts up (decided by the owner file
  holder), so a restart does not double every counter.
- Files of exited processes are purged before each merge for `gauge_live*` metrics,
  so pool gauges do not count dead processes. Counter files are kept, so a process
  exiting never makes a cumulative value go backwards.

### 3.3 Aggregation semantics: sum, not average

prometheus_client has no `avg` multiprocess mode, and that is the right call —
averaging these metrics would almost always be wrong:

| Metric type | Aggregation | Why not an average |
|---|---|---|
| Counter (hit/miss/transfer/evicted/allocation failures) | sum | Cumulative values can only accumulate. For a per-process average, do it in PromQL: `sum(rate(...)) / count(rate(...))` |
| Gauge `mempool_total/free_blocks` | sum (`livesum`) | Their only use is forming a ratio, and **the ratio has to be taken after summing**. Averaging per-process ratios over-weights small pools: a 1000-block pool at 10% and a 100-block pool at 90% is 17.3% full, not 50% |
| Gauge `metrics_server_info` | max | An existence flag that is always 1, not an additive quantity |

Counters therefore get **larger after upgrading** (roughly by the process count).
That is the fix taking effect, not a regression. If nothing changes, the deployment
already had a single process recording metrics.

### 3.4 Two things to watch

1. **MLA H2D is a broadcast** (every rank reads the full KV), so
   `flexkv_py_transfer_bytes_total` sums to TP times the logical KV volume. That is
   physically what crosses PCIe, but reading it as "KV throughput" overstates it.
2. **`mempool_*_blocks` changes meaning** from "one process's pool" to "the whole
   node". Alert thresholds tuned against a single instance need to be re-set. This
   assumes each process owns a disjoint pool; if a deployment shares one pool and
   every process reports the full value, these two gauges must become `livemax`.

### 3.5 Isolation

Two independent FlexKV deployments on the same host sharing a port also share the
aggregation directory. To isolate them:

```bash
export FLEXKV_PY_METRICS_MULTIPROC_DIR=/tmp/flexkv-multiproc-instance-a
```

If the host engine (sglang/vLLM) already runs prometheus_client in multiprocess mode,
FlexKV joins that engine's directory instead of overriding it.

---

## 4. Monitoring Stack Deployment

### 4.1 Directory Structure

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

### 4.2 Quick Deploy

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

### 4.3 Service Access

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

### 4.4 Accessing Grafana Dashboards

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
