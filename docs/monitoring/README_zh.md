# FlexKV Prometheus Metrics 文档

FlexKV 集成了基于 [Prometheus](https://prometheus.io/) 的运行时指标监控框架，覆盖 Python 和 C++ 两层关键路径。该框架以**零侵入**方式嵌入 FlexKV 运行时——用户只需设置环境变量 `FLEXKV_ENABLE_METRICS=1`，即可在应用运行期间自动收集缓存命中、内存池状态、数据传输等核心指标，并通过标准 HTTP 端点暴露给 Prometheus 进行采集和可视化（Grafana）。

---

## 一、配置说明

### 1.1 环境变量

| 环境变量 | 默认值 | 描述 |
|---|---|---|
| `FLEXKV_ENABLE_METRICS` | `0` | 启用指标收集（设为 `1` 启用，默认禁用） |
| `FLEXKV_PY_METRICS_PORT` | `8080` | Python 指标 HTTP 服务端口 |
| `FLEXKV_CPP_METRICS_PORT` | `8081` | C++ 指标 HTTP 服务端口 |
| `FLEXKV_PY_METRICS_MULTIPROC_DIR` | 由端口推导 | Python 指标跨进程聚合目录，详见[第三节](#三多进程聚合python-指标) |

### 1.2 配置方式

```bash
# Enable FlexKV metrics collection
export FLEXKV_ENABLE_METRICS=1

# Custom ports (optional)
export FLEXKV_PY_METRICS_PORT=8080
export FLEXKV_CPP_METRICS_PORT=8081
```

---

## 二、指标总览

### 2.1 Python 运行时指标 (`flexkv_py_*`)

Python 指标由 `GlobalCacheEngine` 在 `cache_engine.py` 中记录，通过 `FlexKVMetricsCollector` 收集。

| 指标名称 | 类型 | 标签 | 描述 |
|---|---|---|---|
| `flexkv_py_cache_hit_blocks_total` | Counter | `device` | 缓存命中的 blocks 总数 |
| `flexkv_py_cache_miss_blocks_total` | Counter | - | 缓存未命中的 blocks 总数（所有层级均未命中） |
| `flexkv_py_transfer_blocks_total` | Counter | `transfer_type`, `operation` | 传输的 blocks 总数 |
| `flexkv_py_transfer_ops_total` | Counter | `transfer_type`, `operation` | 传输操作次数 |
| `flexkv_py_transfer_bytes_total` | Counter | `transfer_type`, `operation` | 传输字节总数 |
| `flexkv_py_mempool_total_blocks` | Gauge | `device` | 内存池总 blocks |
| `flexkv_py_mempool_free_blocks` | Gauge | `device` | 内存池空闲 blocks |
| `flexkv_py_evicted_blocks_total` | Counter | `device` | 驱逐的 blocks 总数 |
| `flexkv_py_allocated_blocks_total` | Counter | `device` | 分配的 blocks 总数 |
| `flexkv_py_allocation_failures_total` | Counter | `mode` | 资源分配失败次数 |
| `flexkv_py_metrics_server_info` | Gauge | - | 恒为 `1`，标记该端口由 FlexKV 指标端点占用 |

> 除 `flexkv_py_metrics_server_info` 外，上表所有 `flexkv_py_*` 指标都是**本节点所有 FlexKV 进程之和**，不是单个进程的值。详见[第三节](#三多进程聚合python-指标)。

---

### 2.2 C++ 运行时指标 (`flexkv_cpp_*`)

C++ 指标由 `MetricsManager` 单例管理，主要在 RadixTree 缓存操作和数据传输中埋点。

| 指标名称 | 类型 | 标签 | 描述 |
|---|---|---|---|
| `flexkv_cpp_cache_ops_total` | Counter | `operation` | RadixTree 缓存操作次数 |
| `flexkv_cpp_cache_blocks_total` | Counter | `operation` | RadixTree 缓存操作涉及的 blocks 数 |

---

## 三、多进程聚合（Python 指标）

> 深入阅读：[设计文档](./multiprocess_metrics_design_zh.md)（问题与方案取舍） ·
> [修复文档](./multiprocess_metrics_fix_zh.md)（改动清单、验证、升级影响）

### 3.1 为什么需要

Python 指标由 `GlobalCacheEngine` 记录（`cache_engine.py` 构造时初始化 collector），
所以"有几个进程写指标" = "有几个进程持有 `GlobalCacheEngine`"。

**sglang 路径下这个数字是 1**：只有 sync leader 会建 `KVManager`
（`connector.py:244-245`；判据 `comm.py:160-162` 的
`pp_rank==0 and tp_rank==0 and cp_rank==0`，类注释称其为 "the unique rank"），
其余 rank 只有 `KVTPClient`，不碰 metrics。`server_client_mode` 下同样只有一个
写方——`KVTaskEngine` 在 `KVServer` 独立子进程里（`server.py:183`）。

既然如此，为什么还要做跨进程聚合：

1. **端口冲突此前是静默的**：抢不到端口的进程 `warning` 后 `return True`，调用方
   以为自己已经在暴露指标。同机跑两套 FlexKV（例如 P/D 分离的两个引擎共用 8080）
   就会命中——第二个部署全程空转且零报错。
2. **第二个写方随时会出现**：传输计时天然产生在 spawn 的 TransferManager 子进程
   里（`transfer_manager.py:803`）。只要把耗时接进 Prometheus，没有聚合就全丢。
   vLLM / TRT-LLM adapter 建 `KVManager` 时没有 sync leader 门控，也是同理。

聚合之前，多写方场景下只有抢到端口的进程能暴露指标，且失败无任何报错。

### 3.2 工作方式

- 所有进程把指标写进同一个 mmap 目录（prometheus_client 的多进程模式），默认路径
  由端口推导：`<tmp>/flexkv-multiproc-<FLEXKV_PY_METRICS_PORT>`。同一次部署的进程
  端口相同，因此共享同一目录。
- **只有主进程**起 HTTP 端点；其余进程只写目录。端点上看到的是合并后的结果。
- 若另一个 FlexKV 进程已经占用了该端口，本进程不再重复绑定，视为共享同一端点。
- 目录在**首次部署启动时清空**（以 owner 文件的持有者为准），因此重启不会让计数翻倍。
- 进程退出后，其 `gauge_live*` 文件会在下次采集时被清理，水位类指标不会把死进程算进去；
  Counter 文件则保留，累计量不会因为某个进程退出而回退。

### 3.3 聚合语义：求和，不是平均

prometheus_client 的多进程模式没有 `avg`，这是合理的——这批指标取平均基本都是错的：

| 指标类型 | 聚合方式 | 说明 |
|---|---|---|
| Counter（命中/未命中/传输/驱逐/分配失败） | 求和 | 累计量只能累加。要看"单进程平均"请在 PromQL 层做：`sum(rate(...)) / count(rate(...))` |
| Gauge `mempool_total/free_blocks` | 求和（livesum） | 这两个指标唯一用途是算水位，而**比值必须先求和再相除**。先算各进程水位再平均，会让小池被等权放大：1000 块的池用了 10%、100 块的池用了 90%，真实水位 17.3%，平均得 50% |
| Gauge `metrics_server_info` | 取最大值 | 恒为 1 的存在性标志，不是可累加量 |

因此**升级后计数类指标会变大**（约等于进程数倍），这是修复生效，不是回归。若升级后
数字没变，说明该部署本来就只有一个进程在记录。

### 3.4 需要注意的两点

1. **MLA 的 H2D 是 broadcast**（每个 rank 各读一份全量 KV），所以
   `flexkv_py_transfer_bytes_total` 求和后会是逻辑 KV 量的 TP 倍。这是物理真值
   （PCIe 上确实过了 N 份），但按"KV 吞吐量"解读会虚高。
2. **`mempool_*_blocks` 的含义从"一个进程的池"变成"全节点之和"**。如果之前拿它当
   单实例水位配告警，阈值需要重设。前提是各进程的池互不相交；如果某部署所有进程
   共享同一个池、各自上报全量，这两个 Gauge 要改成 `livemax`。

### 3.5 隔离

同机运行两套独立 FlexKV 且共用一个端口时，会共享同一聚合目录。需要隔离时显式指定：

```bash
export FLEXKV_PY_METRICS_MULTIPROC_DIR=/tmp/flexkv-multiproc-instance-a
```

若宿主引擎（sglang/vLLM）已经启用了 prometheus_client 的多进程模式，FlexKV 会
沿用引擎的目录，不会覆盖。

---

## 四、监控组件部署说明

### 4.1 目录结构

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

### 4.2 快速部署

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

### 4.3 访问服务

| 服务 | 地址 | 说明 |
|---|---|---|
| Python Metrics | `http://localhost:8080/metrics` | Python 运行时指标端点 |
| C++ Metrics | `http://localhost:8081/metrics` | C++ 运行时指标端点 |
| Prometheus | `http://localhost:9090` | 指标查询界面 |
| Grafana | `http://localhost:3000` | 可视化仪表板 |

**快速验证指标端点：**

```bash
# Verify Python metrics endpoint
curl -s http://localhost:8080/metrics | grep flexkv_py_

# Verify C++ metrics endpoint
curl -s http://localhost:8081/metrics | grep flexkv_cpp_
```

### 4.4 访问 Grafana 仪表板

1. 打开浏览器访问 `http://localhost:3000`
2. 使用默认账号登录：用户名 `admin`，密码 `admin`
3. 进入 **Dashboards → FlexKV Demo** 查看预置仪表板

**预置仪表板包含以下典型面板：**

| 分区 | 面板 | 说明 |
|---|---|---|
| Python Runtime Metrics | Cache Hit/Miss Rate | 缓存命中/未命中速率 |
| Python Runtime Metrics | Memory Pool Blocks | 内存池块数统计 |
| Python Runtime Metrics | Transfer Throughput | 数据传输吞吐量 |
| C++ Runtime Metrics | Cache Operations Rate | 缓存操作速率 |
| C++ Runtime Metrics | Cache Blocks Rate | 缓存块操作速率 |

> 用户可以按需创建自定义面板并添加和配置 PromQL 查询语句。
