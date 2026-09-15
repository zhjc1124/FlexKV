"""FlexKV metric registries and cross-process aggregation.

FlexKV records metrics from more than one process: ``GlobalCacheEngine`` lives
in the engine process, while ``TransferEngine`` - and therefore all transfer
timing - runs in the spawned TransferManager subprocess. Without aggregation
only the process that happens to bind the metrics port is visible and the rest
is silently dropped.

``prometheus_client`` merges across processes only when
PROMETHEUS_MULTIPROC_DIR is set *before* the first metric object is built, so
this module must be imported before anything else imports
``prometheus_client``.

Two registries, and the split matters:

- ``METRIC_REGISTRY``  where the metric objects live; never served.
- ``serve_registry()`` a fresh registry holding only ``MultiProcessCollector``;
  this is what the HTTP endpoint exposes.

A registered metric object reports its own process-local value while
``MultiProcessCollector`` reports the merged value of every process, so serving
both out of one registry emits every series twice.
"""

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
    """Drop mmap files left behind by processes that are gone.

    ``MultiProcessCollector`` reads every file in the directory, so files of
    dead processes would keep contributing stale values forever.
    """
    try:
        entries = os.listdir(path)
    except OSError:
        return
    for name in entries:
        if not name.endswith(".db"):
            continue
        # Files are named <type>[_<mode>]_<pid>.db, e.g. counter_1234.db.
        pid = name[:-3].rsplit("_", 1)[-1]
        if not pid.isdigit() or _pid_alive(int(pid)):
            continue
        try:
            os.remove(os.path.join(path, name))
        except FileNotFoundError:
            pass


def _prepare_multiproc_dir() -> str:
    path = _resolve_multiproc_dir()
    os.environ.setdefault(_MULTIPROC_DIR_ENV, path)
    path = os.environ[_MULTIPROC_DIR_ENV]
    os.makedirs(path, exist_ok=True)
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

_serve_registry = None
_serve_lock = threading.Lock()


def serve_registry():
    """Registry exposed on /metrics: the merged view of every FlexKV process."""
    global _serve_registry
    with _serve_lock:
        if _serve_registry is None:
            if PROMETHEUS_AVAILABLE and MULTIPROC_DIR is not None:
                registry = CollectorRegistry()
                MultiProcessCollector(registry, path=MULTIPROC_DIR)
                _serve_registry = registry
            else:
                _serve_registry = METRIC_REGISTRY
        return _serve_registry
