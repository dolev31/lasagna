"""Document Lifecycle Management for Lasagna.

Public API::

    from lasagna.lifecycle import (
        DocumentLifecycleManager,
        DocumentExpiredError,
        GarbageCollector,
        LifecyclePolicy,
        LifecycleScheduler,
        OrphanDetector,
    )

    # Results / report types
    from lasagna.lifecycle import (
        GCReport,
        GCPhaseResult,
        ConsistencyReport,
        RepairReport,
        BulkDeleteResult,
    )
"""

from ._results import (
    BulkDeleteResult,
    ConsistencyReport,
    GCPhaseResult,
    GCReport,
    OrphanRecord,
    RepairReport,
)
from .gc import GarbageCollector
from .manager import DocumentExpiredError, DocumentLifecycleManager
from .orphan_detector import OrphanDetector
from .policy import LifecyclePolicy
from .scheduler import LifecycleScheduler

__all__ = [
    # Primary entry points
    "DocumentLifecycleManager",
    "DocumentExpiredError",
    "GarbageCollector",
    "LifecyclePolicy",
    "LifecycleScheduler",
    "OrphanDetector",
    # Result / report types
    "GCReport",
    "GCPhaseResult",
    "ConsistencyReport",
    "RepairReport",
    "BulkDeleteResult",
    "OrphanRecord",
]
