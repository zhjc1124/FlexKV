"""FlexKV metric registries and cross-process aggregation.

FlexKV may record metrics from more than one process. The collector is built
in ``GlobalCacheEngine.__init__`` (``flexkv/cache/cache_engine.py``), and how
many processes hold one depends on the integration:

- sglang: exactly one. Only the sync leader builds a ``KVManager``
  (``integration/sglang/connector.py``; the gate is ``pp_rank == 0 and
  attn_tp_rank == 0 and attn_cp_rank == 0`` in ``integration/sglang/comm.py``).
  Under ``server_client_mode`` the single ``KVTaskEngine`` lives in the
  ``KVServer`` subprocess instead.
- vLLM / TRT-LLM and multi-instance hosts: potentially several, because those
  adapters build a ``KVManager`` without a leader gate.
- The spawned TransferManager subprocess becomes a second writer the moment
  transfer durations are exported.

Without aggregation, any writer that does not win the bind is silently
dropped, and losing the bind used to report success anyway.

``prometheus_client`` merges across processes only when
PROMETHEUS_MULTIPROC_DIR is set *before* the first metric object is built, so
this module must be imported before anything else imports
``prometheus_client``.

Two registries, and the split matters:

- ``METRIC_REGISTRY``  where the metric objects live; never served.
- ``serve_registry()`` a fresh registry holding only the process collector;
  this is what the HTTP endpoint exposes.

A registered metric object reports its own process-local value while
``MultiProcessCollector`` reports the merged value of every process, so serving
both out of one registry emits every series twice.
"""

import contextlib
import os
import tempfile
import threading

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger

logger = flexkv_logger

_MULTIPROC_DIR_ENV = "PROMETHEUS_MULTIPROC_DIR"
_MULTIPROC_DIR_OVERRIDE_ENV = "FLEXKV_PY_METRICS_MULTIPROC_DIR"


def _resolve_multiproc_dir() -> str:
    """Directory shared by every FlexKV process of one deployment.

    Keyed by the metrics port when not overridden: all FlexKV processes of a
    deployment point at the same port, so they share the aggregated view
    behind it.
    """
    override = os.environ.get(_MULTIPROC_DIR_OVERRIDE_ENV)
    if override:
        return override
    return os.path.join(
        tempfile.gettempdir(),
        f"flexkv-multiproc-{GLOBAL_CONFIG_FROM_ENV.py_metrics_port}",
    )


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by another user: alive, we just cannot signal it.
        return True
    return True


def _purge_dead_process_files(path: str) -> None:
    """Drop ``gauge_live*`` files left behind by processes that are gone.

    Only live gauges are purged, and that is the point:

    - counters are cumulative - a process that dies still transferred those
      blocks, so its file has to stay;
    - a live gauge is a snapshot ("how full is my pool right now") - once the
      process is gone there is nothing to report, and keeping its file would
      inflate every sum and every ratio derived from it.

    ``livesum`` and ``sum`` take the same branch in the merge, so without this
    purge ``livesum`` behaves exactly like ``sum``. prometheus_client only
    removes these files when something calls ``mark_process_dead``, which
    nothing in FlexKV does.
    """
    try:
        entries = os.listdir(path)
    except OSError:
        return
    for name in entries:
        if not name.startswith("gauge_live") or not name.endswith(".db"):
            continue
        # Files are named gauge_<mode>_<pid>.db, e.g. gauge_livesum_1234.db.
        pid = name[:-3].rsplit("_", 1)[-1]
        if not pid.isdigit() or _pid_alive(int(pid)):
            continue
        try:
            os.remove(os.path.join(path, name))
        except FileNotFoundError:
            pass


def _purge_all_files(path: str) -> None:
    """Empty the directory. Used when a fresh deployment claims it."""
    try:
        entries = os.listdir(path)
    except OSError:
        return
    for name in entries:
        if name.endswith(".db"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(path, name))


_OWNER_FILE = ".flexkv-owner"


def _claim_directory(path: str) -> bool:
    """True when this process is the first of a fresh deployment.

    prometheus_client requires the multiprocess directory to be emptied at
    startup. Files are named after pid, so a restart would otherwise stack new
    files next to the previous run's and every counter would come back doubled.

    Whoever creates the owner file wins and wipes the directory; later
    processes of the same deployment just join it. An owner file whose pid is
    gone belongs to a dead deployment and is taken over.
    """
    sentinel = os.path.join(path, _OWNER_FILE)
    for _ in range(2):
        try:
            fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            return True
        except FileExistsError:
            pass
        try:
            with open(sentinel) as f:
                owner = int(f.read().strip() or 0)
        except (OSError, ValueError):
            owner = 0
        if owner and owner != os.getpid() and _pid_alive(owner):
            return False
        with contextlib.suppress(FileNotFoundError):
            os.remove(sentinel)
    return False


def _prepare_multiproc_dir() -> str:
    path = _resolve_multiproc_dir()
    # setdefault, not assignment: a host engine that already runs in
    # multiprocess mode keeps its own directory and FlexKV joins it.
    os.environ.setdefault(_MULTIPROC_DIR_ENV, path)
    path = os.environ[_MULTIPROC_DIR_ENV]
    os.makedirs(path, exist_ok=True)
    if _claim_directory(path):
        _purge_all_files(path)
    else:
        _purge_dead_process_files(path)
    return path


# Only touch the environment when metrics are on, so that a disabled FlexKV
# never switches the host engine's prometheus_client into multiprocess mode.
_MULTIPROC_ENABLED = GLOBAL_CONFIG_FROM_ENV.enable_metrics
MULTIPROC_DIR = _prepare_multiproc_dir() if _MULTIPROC_ENABLED else None

# prometheus_client picks its value backend at import time, so the directory
# has to be in the environment before this import.
try:
    from prometheus_client import CollectorRegistry
    from prometheus_client import values as _prom_values
    from prometheus_client.multiprocess import MultiProcessCollector

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

if (
    PROMETHEUS_AVAILABLE
    and MULTIPROC_DIR is not None
    and _prom_values.ValueClass.__name__ != "MmapedValue"
):
    # The host engine imported prometheus_client before FlexKV could set
    # PROMETHEUS_MULTIPROC_DIR, so the single-process backend is already bound.
    _prom_values.ValueClass = _prom_values.MultiProcessValue()
    logger.warning(
        "[FlexKV PyMetrics] prometheus_client was imported before FlexKV could enable "
        "multiprocess mode; switching the value backend now. Metrics created by the "
        "host engine after this point will use the multiprocess backend too."
    )

METRIC_REGISTRY = CollectorRegistry() if PROMETHEUS_AVAILABLE else None

if PROMETHEUS_AVAILABLE:

    class FlexKVMultiProcessCollector(MultiProcessCollector):
        """``MultiProcessCollector`` that drops dead processes before merging.

        ``livesum`` and ``sum`` take the same branch in ``_accumulate_metrics``
        - both are a plain ``+=``. The `live` prefix only decides whether
        ``mark_process_dead(pid)`` would delete the file, and nothing in
        FlexKV calls it, so a process that dies leaves its gauge value on disk
        and it keeps counting forever. Purging before every merge is what
        actually makes ``livesum`` mean "live".

        The cost is one ``listdir`` plus one ``kill(pid, 0)`` per file, against
        a handful of processes.
        """

        def collect(self):
            if MULTIPROC_DIR is not None:
                _purge_dead_process_files(MULTIPROC_DIR)
            return super().collect()

else:

    class FlexKVMultiProcessCollector:  # pragma: no cover - prometheus_client missing
        def __init__(self, *args, **kwargs):
            raise RuntimeError("prometheus_client is not installed")


_serve_registry = None
_serve_lock = threading.Lock()


def serve_registry():
    """Registry exposed on /metrics: the merged view of every FlexKV process."""
    global _serve_registry
    with _serve_lock:
        if _serve_registry is None:
            if PROMETHEUS_AVAILABLE and MULTIPROC_DIR is not None:
                registry = CollectorRegistry()
                FlexKVMultiProcessCollector(registry, path=MULTIPROC_DIR)
                _serve_registry = registry
            else:
                _serve_registry = METRIC_REGISTRY
        return _serve_registry
