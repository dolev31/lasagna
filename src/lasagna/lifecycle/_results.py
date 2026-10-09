"""Pure data classes for GC and orphan detection results.

These are deliberately free of business logic — they only carry data so that
callers (REST handlers, tests, audit log writers) can inspect outcomes without
importing the GC engine itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class GCPhaseResult:
    """Result of a single GC phase."""

    phase: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    docs_scanned: int = 0
    docs_cleaned: int = 0
    bytes_freed: int = 0
    errors: int = 0
    details: Dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000

    def to_dict(self) -> Dict:
        return {
            "phase": self.phase,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "docs_scanned": self.docs_scanned,
            "docs_cleaned": self.docs_cleaned,
            "bytes_freed": self.bytes_freed,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "details": self.details,
        }


@dataclass
class GCReport:
    """Aggregated result of a full GC cycle (all phases)."""

    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    phases: List[GCPhaseResult] = field(default_factory=list)
    aborted: bool = False
    abort_reason: Optional[str] = None

    @property
    def total_docs_cleaned(self) -> int:
        return sum(p.docs_cleaned for p in self.phases)

    @property
    def total_bytes_freed(self) -> int:
        return sum(p.bytes_freed for p in self.phases)

    @property
    def total_errors(self) -> int:
        return sum(p.errors for p in self.phases)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "total_docs_cleaned": self.total_docs_cleaned,
            "total_bytes_freed": self.total_bytes_freed,
            "total_errors": self.total_errors,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "phases": [p.to_dict() for p in self.phases],
        }


@dataclass
class OrphanRecord:
    """A single orphaned document detected by OrphanDetector."""

    document_id: str
    # Where the orphan was found:
    in_db: bool  # True  → DB record exists
    in_vector: bool  # True  → chunks exist in vector backend

    @property
    def kind(self) -> str:
        if self.in_db and not self.in_vector:
            return "db_only"  # Processing failure — no vectors
        if self.in_vector and not self.in_db:
            return "vector_only"  # Leaked vectors after DB deletion
        return "unknown"


@dataclass
class ConsistencyReport:
    """Result of an orphan detection scan."""

    scanned_at: datetime
    session_id: Optional[str]
    db_count: int
    vector_count: int
    orphans: List[OrphanRecord] = field(default_factory=list)
    # Truncation indicator — set when the vector scan hit vector_scan_limit.
    # Results may be incomplete; increase limit or scope by session_id.
    scan_limited: bool = False
    vector_scan_limit: int = 50_000
    estimated_unscanned: Optional[int] = None

    @property
    def db_only_count(self) -> int:
        return sum(1 for o in self.orphans if o.kind == "db_only")

    @property
    def vector_only_count(self) -> int:
        return sum(1 for o in self.orphans if o.kind == "vector_only")

    @property
    def is_consistent(self) -> bool:
        return len(self.orphans) == 0

    def to_dict(self) -> Dict:
        return {
            "scanned_at": self.scanned_at.isoformat(),
            "session_id": self.session_id,
            "db_count": self.db_count,
            "vector_count": self.vector_count,
            "orphan_count": len(self.orphans),
            "db_only_count": self.db_only_count,
            "vector_only_count": self.vector_only_count,
            "is_consistent": self.is_consistent,
            "scan_limited": self.scan_limited,
            "vector_scan_limit": self.vector_scan_limit,
            "estimated_unscanned": self.estimated_unscanned,
            "orphans": [{"document_id": o.document_id, "kind": o.kind} for o in self.orphans],
        }


@dataclass
class RepairReport:
    """Result of an orphan repair operation."""

    dry_run: bool
    repaired_db_only: int = 0  # DB-only orphans deleted from DB
    repaired_vector_only: int = 0  # Vector-only orphans deleted from vector backend
    errors: int = 0

    def to_dict(self) -> Dict:
        return {
            "dry_run": self.dry_run,
            "repaired_db_only": self.repaired_db_only,
            "repaired_vector_only": self.repaired_vector_only,
            "errors": self.errors,
        }


@dataclass
class BulkDeleteResult:
    """Result of a bulk document deletion."""

    session_id: Optional[str]
    requested: int
    deleted: int
    errors: int = 0
    bytes_freed: int = 0
    chunks_freed: int = 0

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "requested": self.requested,
            "deleted": self.deleted,
            "errors": self.errors,
            "bytes_freed": self.bytes_freed,
            "chunks_freed": self.chunks_freed,
        }
