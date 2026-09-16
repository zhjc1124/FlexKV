# Python 指标跨进程聚合 —— 修复文档

> 分支：`fix/metrics-multiprocess-aggregation`，基线 `origin/main`（`4e29961ee`）
> 配套文档：[设计文档](./multiprocess_metrics_design_zh.md)（问题是什么、为什么这么修）

## 0. 结论先行

> **前提更正**：第一版这里写"TP=8 只看到 1/8"是错的。sglang 路径下
> `GlobalCacheEngine` 全局唯一（只有 sync leader 建 `KVManager`，
> `comm.py:160-162`），Python 指标**本来就只有一个写方，覆盖率 100%**。
> 详见[设计文档 §1](./multiprocess_metrics_design_zh.md#1-背景指标在哪里产生)。

这次改动的实际收益分三块：

| 项 | 修复前 | 修复后 |
|---|---|---|
| 端口被**非 FlexKV** 服务占用 | warning 后 `return True`，端点其实是别人的 | `raise`，日志显式报错（不阻断启动） |
| 端口被**另一个 FlexKV** 占用 | 同样 `return True`，第二个部署全程空转 | 识别为共享，走聚合目录，两边数据都可见 |
| 第二个写方出现时（TransferManager 埋点 / 非 sglang 多引擎 / 多实例同机） | 全部丢失且无报错 | 自动合并 |

| 指标 | 单写方部署（sglang 现状） | 多写方部署 |
|---|---|---|
| `cache_hit/miss_blocks_total` 等 Counter | **数值不变** | 所有进程求和 |
| `mempool_total/free_blocks` | **数值不变** | `livesum` = 存活进程求和 |
| `metrics_server_info` | 新增，恒为 1，用于识别端口归属 | 同左 |

**所以：sglang 单实例部署升级后数值不应有任何跳变。** 如果计数跳到 N 倍，说明
该部署本来就有 N 个进程在写（多实例同机、或 vLLM/TRT-LLM 路径），那是修复生效，
不是回归。

---

## 1. 范围

只做跨进程聚合一条线。以下**均不在**本次改动内：

- 传输耗时 Histogram（计时源在 `transfer/trace.py`，独立项）
- 分层语义（L2/L3/L4 归属、层间 block 流）
- 健康层（in-flight / backlog / IO 错误）
- C++ 指标（`flexkv_cpp_*`，端口 8081，编译期开关）
- 任何新指标、新标签、埋点语义变更

除 `flexkv_py_metrics_server_info` 外没有新增指标，也没有删指标。

---

## 2. 改动清单

```
docs/monitoring/multiprocess_metrics_design_zh.md   | 251 +++  新增
docs/monitoring/multiprocess_metrics_fix_zh.md     |  本文档
docs/monitoring/README_zh.md / README_en.md        |  聚合语义章节 + 环境变量
flexkv/metrics/registry.py                         | 237 +++  新增
flexkv/metrics/__init__.py                         |   5 +
flexkv/metrics/collector.py                        |  70 +-
flexkv/metrics/server.py                           |  82 +-
```

### 2.1 `flexkv/metrics/registry.py`（新增）

整个修复的地基。必须在 `prometheus_client` 被 import 之前执行，因为它要在那时把
`PROMETHEUS_MULTIPROC_DIR` 放进环境——prometheus_client 在 import 时就绑定值后端，
之后再设环境变量无效。

| 函数 | 作用 |
|---|---|
| `_resolve_multiproc_dir()` | 默认 `<tmpdir>/flexkv-multiproc-<port>`，可用 `FLEXKV_PY_METRICS_MULTIPROC_DIR` 覆盖 |
| `_pid_alive()` | `kill(pid, 0)` 判活；`PermissionError` 视为存活（属主是别的用户） |
| `_purge_dead_process_files()` | 删除**死进程的 `gauge_live*` 文件**。只清 live gauge、不清 Counter，理由见设计文档 4.6 |
| `_purge_all_files()` | 清空目录，仅部署首次启动时调用 |
| `_claim_directory()` | owner 文件裁定"谁是本次部署第一个进程"，决定清空还是加入 |
| `FlexKVMultiProcessCollector` | 每次 merge 前先清死进程文件，让 `livesum` 真的等于"活着" |
| `METRIC_REGISTRY` / `serve_registry()` | 两个注册表，见 2.3 |

两个非显然的判断：

1. **`_claim_directory` 是必需的，不是防护性代码。** 文件名按 pid 命名
   （`counter_1234.db`），重启后 pid 变了就是新文件；不清空的话合并结果是
   旧值 + 新值。实测：同一目录连续跑两次，不清空时 `cache_hit_blocks_total`
   从 10 变成 20。
2. **只清 `gauge_live*`、不清 Counter。** Counter 是累计量，进程死了它搬过的
   blocks 也是事实，删掉会让曲线回退；live Gauge 是快照，进程没了就不该有数。

### 2.2 `flexkv/metrics/__init__.py`

```python
# Must come first: it puts PROMETHEUS_MULTIPROC_DIR in the environment before
# prometheus_client is imported, ...
from flexkv.metrics import registry
```

一行加在文件最顶部。**import 顺序是功能的一部分**，不要挪。

### 2.3 `flexkv/metrics/collector.py`

| 改动 | 说明 |
|---|---|
| `import os` → `import multiprocessing` | 用于判断是否主进程 |
| 顶部 `from flexkv.metrics.registry import METRIC_REGISTRY` | 放在 `from prometheus_client import ...` 之前 |
| 所有 metric 加 `registry=METRIC_REGISTRY` | 从"进程内默认 REGISTRY"迁到不可服务的注册表 |
| `mempool_*` Gauge 无条件 `multiprocess_mode="livesum"` | 原来只在 `os.environ.get("PROMETHEUS_MULTIPROC_DIR")` 为真时才设，而那个条件在 FlexKV 自己管理目录前**永远为假**（D4） |
| `_auto_start_metrics_server()`：非主进程提前返回 | 只有 `current_process().name == "MainProcess"` 的进程起端点 |
| `_auto_start_metrics_server()`：`RuntimeError` → error 日志 | 端口被外来服务占用时不再是一行 warning |

C++ 指标的配置调用（`_configure_cpp_metrics()`）保持原样，未改动其行为。

### 2.4 `flexkv/metrics/server.py`

| 改动 | 修复前 | 修复后 |
|---|---|---|
| 服务端点用的注册表 | `registry=REGISTRY`（进程内默认） | `registry=serve_registry()`（合并视图） |
| 端口被非 FlexKV 服务占用 | warn + `return False`，调用方无感知 | **抛 `RuntimeError`**，提示换端口 |
| 端口被另一个 FlexKV 进程占用 | warn + `_server_started = True` + `return True` | 视为共享，info 日志 + `return True`（行为不变，语义写清） |
| 其他 bind 失败 | warn/error + `return False` | **抛 `RuntimeError`** |
| `flexkv_py_metrics_server_info` | 无 | 新增恒为 1 的 Gauge，让端口归属可判定 |

为什么要新增 `metrics_server_info`：其余指标都只在被观测后才出现，刚启动的
FlexKV 端点在 `/metrics` 上是空的，无法与任意 Prometheus exporter 区分。而
"端口上是不是 FlexKV"正是决定共享还是报错的依据。

> 该 Gauge 没有 `pid` 标签：合并步骤会剥掉所有非 `all`/`liveall` 模式 gauge 的
> `pid` 标签（prometheus_client `multiprocess.py`），加了只在非聚合场景出现，
> 同一指标两种形状。

---

## 3. 验证

### 3.1 本机隔离验证（Mac，无 CUDA）

仓库 `conftest.py` 一 import 就 dlopen CUDA 挂掉，属本机固有限制。用 stub 挡掉
`flexkv.common.config` / `debug` 后跑真实代码路径：

| 验证项 | 结果 |
|---|---|
| `prometheus_client` 值后端 | `MmapedValue`（多进程后端已激活） |
| 跨进程 Counter 合并 | 主进程 3 + 子进程 7 = **10** ✅ |
| `livesum` 排除死进程 | 子进程死后 `mempool_total_blocks` 从 3072 回到 **1024** ✅ |
| Counter 不因进程退出回退 | 子进程死后 `cache_hit_blocks_total` 仍为 **10** ✅ |
| 只有主进程服务端点 | 子进程 `is_server_running() == False` ✅ |
| 无重复 series | 8 个指标名各 1 条 ✅ |
| 同注册表确实会重复（设计依据） | `fx_dup_total` 出现 **2 次** ✅ |
| 重启不翻倍 | 连续跑两次，第二次 `cache_hit_blocks_total` 仍为 **10** ✅ |
| 端口被外来服务占用 | 抛 `RuntimeError`：*Port ... is occupied by a service that is not a FlexKV metrics endpoint* ✅ |
| 端口被另一个 FlexKV 占用 | `start_metrics_server()` 返回 `True`，共享 ✅ |

### 3.2 上机验证（GPU 机）

```bash
export FLEXKV_ENABLE_METRICS=1

# 启动引擎后：
# 1) 有几个进程在记录 —— 每个 counter_<pid>.db 对应一个
ls ${FLEXKV_PY_METRICS_MULTIPROC_DIR:-/tmp/flexkv-multiproc-8080}

# 2) 端点可达且能自证身份
curl -s http://127.0.0.1:8080/metrics | grep flexkv_py_metrics_server_info
# 期望: flexkv_py_metrics_server_info 1.0

# 3) 计数已合并 —— 与进程数对比
curl -s http://127.0.0.1:8080/metrics | grep flexkv_py_cache_hit_blocks_total
```

对比升级前后同 workload 的 `cache_hit_blocks_total`：

- **数字没变** → 该部署本来就只有 1 个写方（sglang 单实例正是如此），符合预期。
- **跳到约 N 倍** → 该部署有 N 个进程在写（多实例同机、或 vLLM / TRT-LLM 路径），
  修复生效，不是回归。

### 3.3 `livesum` 正确性判据（需要你确认）

`livesum` 正确的前提是**各进程的池互不相交**。代码里无法断定（`Mempool` 是纯 id
分配器，`flexkv/cache/mempool.py:8-19`，不持有物理内存），上机看一眼：

```
启动日志里的 num_cpu_blocks = X
flexkv_py_mempool_total_blocks{device="cpu"} = ?
```

- **等于 X** → N 个进程分摊了配置总量，池独立，`livesum` 正确
- **等于 N×X** → 每个进程都报了全量，池共享，这两个 Gauge 要改成 `livemax`

---

## 4. 升级后的影响

### 4.1 数字会怎么变

| 场景 | 变化 |
|---|---|
| 单进程部署 | 无变化（聚合目录只有一个进程） |
| 多进程部署 | Counter 约 ×N；Gauge `mempool_*` 从单进程变为求和 |
| 已有告警基于 `mempool_*` 单实例水位 | **阈值需重设** |
| 已有面板用 `flexkv_py_transfer_bytes_total` 当"KV 吞吐" | MLA 下会因 broadcast 虚高 N 倍，见 4.2 |

### 4.2 两个需要写进口径的副作用

1. **MLA 的 H2D 是 broadcast**（每个 rank 各读一份全量 KV），所以
   `flexkv_py_transfer_bytes_total` 求和后是逻辑 KV 量的 N 倍。这是物理真值
   （PCIe 上确实过了 N 份），但按"KV 吞吐量"解读会虚高。以前只看 1 个进程反而
   "看起来对"。
2. **`mempool_*_blocks` 含义变化**：从"一个进程的池"变成"全节点之和"。

两者都已在 `README_zh.md` / `README_en.md` 的「聚合语义」章节标注。

### 4.3 不会变的东西

- 指标名、标签名、埋点位置：一个都没动。
- `FLEXKV_ENABLE_METRICS=0` 时：完全不动环境变量、不建目录、不起服务。
- 引擎启动流程：新增的 `RuntimeError` 被捕获成 error 日志，**不会打断启动**。
- C++ 指标：未改动。

### 4.4 性能

可忽略。prometheus_client 的多进程锁是**进程内一把 `threading.Lock`，没有文件锁**；
每次采集多出的成本是一次 `listdir` + 每文件一次 `kill(pid, 0)`，对十几个进程可忽略。

---

## 5. 回滚

改动集中在 `flexkv/metrics/` 且未改任何埋点，直接 revert 分支即可：

```bash
git revert <commit>          # 或整体回退到 origin/main
```

回滚后指标回到"只有抢到端口的进程可见"的口径，无数据损坏（聚合目录在 `/tmp`，
部署启动时会清空）。单写方部署回滚前后数值一致。

只想关掉聚合、保留其他行为（不需要，仅说明）：设 `FLEXKV_ENABLE_METRICS=0`
会连指标一起关掉，没有单独关闭聚合的开关。

---

## 6. 已知限制

| 项 | 说明 |
|---|---|
| PID 复用 | 死进程判定用 `kill(pid, 0)`，pid 被复用会误判存活，残留旧值。低概率，只影响重启后首个采集周期 |
| 同机多部署 | 共用同一端口会共享聚合目录，用 `FLEXKV_PY_METRICS_MULTIPROC_DIR` 隔离 |
| 宿主引擎先 import prometheus_client | 强制切值后端并打 warning；引擎之后建的 metric 也会落进该目录 |
| 冷启动竞态 | 两个进程同时首次启动时，先 claim 的清空目录可能抹掉另一个刚写的值。窗口微秒级，此时值接近 0 |
| 目录残留 | 聚合目录不会自动删除，每次部署启动清空；`/tmp` 自身有清理策略 |

---

## 7. 后续（独立项，本次未做）

- **传输耗时 Histogram**：计时源已在 `transfer/trace.py`（`FLEXKV_ENABLE_TRANSFER_TRACE`），
  只是没接进 Prometheus。这是"零 Histogram"缺口的第一个该补的。
- **分层语义**：L2 CPU / L3 SSD / L4 remote 的归属与层间 block 流。
- **健康层**：in-flight / backlog gauge、IO 错误计数。
- **C++ 指标开关**：`FLEXKV_ENABLE_MONITORING` 目前是编译期决定，装完再 export 无效且失败静默。
