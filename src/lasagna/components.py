"""Add a technique in one line: register a plain function as a pipeline stage.

Write the technique as an ordinary function and put one decorator above it::

    import lasagna

    @lasagna.reranker("keyword_overlap")
    def keyword_overlap(query: str, text: str) -> float:
        return len(set(query.lower().split()) & set(text.lower().split()))

That is the whole integration: the function is wrapped in the stage's interface
(``BaseReranker`` here), registered under the name, and runs inside the full
pipeline wherever a reranker can go. Or name it straight from a sweep, with no
registration and no packaging, as ``module:function``::

    pipeline:
      rerankers: [cross_encoder, "my_ideas:keyword_overlap"]

What each stage's function receives and returns:

==================  ==================================================  ==========================
Stage               Function                                            Returns
==================  ==================================================  ==========================
reranker            ``f(query, text)`` scores one passage               ``float``
reranker            ``f(query, documents, top_k)`` orders them all      ``list[ScoredDocument]``
query_enhancer      ``f(query)``                                        ``list[str]`` (or ``str``)
chunker             ``f(text)``                                         ``list[str]``
embedder            ``f(texts)``                                        one vector per text
retriever           ``f(query, top_k)``                                 ``list[str]`` or documents
document_processor  ``f(file_path)``                                    ``str``
==================  ==================================================  ==========================

A class that already implements the stage's interface can be registered the
same way. The wrapper hands the function copies of its inputs, so a technique
cannot change what another stage sees, whatever it does to them.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import inspect
import marshal
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .core.base_components import (
    BaseChunker,
    BaseDocumentProcessor,
    BaseEmbedder,
    BaseQueryEnhancer,
    BaseReranker,
    BaseRetriever,
    Chunk,
    ProcessedDocument,
    ScoredDocument,
)

_STAGE_ABCS = {
    "reranker": BaseReranker,
    "query_enhancer": BaseQueryEnhancer,
    "chunker": BaseChunker,
    "embedder": BaseEmbedder,
    "retriever": BaseRetriever,
    "document_processor": BaseDocumentProcessor,
}


def _positional(fn: Callable) -> List[inspect.Parameter]:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return []
    return [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]


#: How a reranker function says which shape it has: by the name of its second
#: parameter, or by taking ``top_k``.
_LISTWISE_NAMES = frozenset({"documents", "docs", "passages", "candidates", "results", "hits"})
_POINTWISE_NAMES = frozenset({"text", "passage", "document", "doc", "content", "chunk"})


def _is_listwise(fn: Callable) -> bool:
    """Whether a reranker function orders the list (True) or scores one text.

    Read from the parameter names first, so ``f(query, documents, top_k=5)``
    (the interface's own signature) is listwise although ``top_k`` has a
    default. A third parameter with a default and an unrecognized second name
    is ambiguous, and refused rather than guessed.
    """
    params = _positional(fn)
    names = [p.name for p in params]
    if "top_k" in names or (len(names) > 1 and names[1] in _LISTWISE_NAMES):
        return True
    if len(names) > 1 and names[1] in _POINTWISE_NAMES:
        return False
    required = sum(1 for p in params if p.default is p.empty)
    if required >= 3:
        return True
    if len(params) >= 3:
        raise TypeError(
            f"Cannot tell whether {_identity(fn)} scores one passage or orders the list. "
            f"Name its parameters (query, text) to score one passage, or "
            f"(query, documents, top_k) to order them all."
        )
    return False


def _identity(fn: Callable) -> str:
    return f"{getattr(fn, '__module__', '?')}:{getattr(fn, '__qualname__', repr(fn))}"


_SIMPLE = (int, float, str, bytes, bool, type(None))


def _code_digest(fn: Callable) -> str:
    """A digest of what the function computes: its code and simple captured values.

    Two lambdas, or two functions one factory made with different arguments,
    share ``module:qualname``; the search cache must still tell them apart, in
    this process and in a shared cache across processes.
    """
    h = hashlib.sha256()
    code = getattr(fn, "__code__", None)
    if code is not None:
        try:
            h.update(marshal.dumps(code))
        except ValueError:
            h.update(repr(code).encode())
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:  # an empty cell
            value = None
        h.update((repr(value) if isinstance(value, _SIMPLE) else type(value).__qualname__).encode())
    for value in getattr(fn, "__defaults__", None) or ():
        h.update((repr(value) if isinstance(value, _SIMPLE) else type(value).__qualname__).encode())
    return h.hexdigest()[:12]


class _FunctionComponent:
    """Shared by every wrapper: the function, its registered name, its identity."""

    _fn: Callable
    _registered_name: str = ""

    @property
    def name(self) -> str:
        return self._registered_name or self.__class__.__name__

    @property
    def model_name(self) -> str:
        # Read for cache keys and provenance, so two functions never share one:
        # the registered name, where the function lives, and what it computes.
        return f"{self._registered_name}={_identity(self._fn)}#{_code_digest(self._fn)}"


class FunctionReranker(_FunctionComponent, BaseReranker):
    """A reranker from ``f(query, text) -> float`` or ``f(query, documents, top_k)``."""

    _listwise: bool = False

    def rerank(
        self, query: str, documents: List[ScoredDocument], top_k: int = 5, **kwargs
    ) -> List[ScoredDocument]:
        docs = copy.deepcopy(list(documents))
        if self._listwise:
            out = list(self._fn(query, docs, top_k))
        else:
            for d in docs:
                d.score = float(self._fn(query, d.content))
            out = sorted(docs, key=lambda d: d.score, reverse=True)
        return out[:top_k]


class FunctionQueryEnhancer(_FunctionComponent, BaseQueryEnhancer):
    """A query enhancer from ``f(query) -> list[str] | str``."""

    def enhance(self, query: str, **kwargs) -> List[str]:
        out = self._fn(query)
        variants = [out] if isinstance(out, str) else list(out or [])
        return [query] + [v for v in variants if isinstance(v, str) and v and v != query]


class FunctionChunker(_FunctionComponent, BaseChunker):
    """A chunker from ``f(text) -> list[str]``; offsets are found in the text."""

    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        chunks: List[Chunk] = []
        cursor = 0
        for i, piece in enumerate(self._fn(text) or []):
            if isinstance(piece, Chunk):
                chunks.append(copy.deepcopy(piece))
                continue
            piece = str(piece)
            start = text.find(piece, cursor)
            start = cursor if start < 0 else start
            cursor = start + len(piece)
            chunks.append(Chunk(content=piece, chunk_index=i, start_char=start, end_char=cursor))
        return chunks


class FunctionEmbedder(_FunctionComponent, BaseEmbedder):
    """An embedder from ``f(texts) -> one vector per text``."""

    _dimension: Optional[int] = None

    def encode(self, texts: List[str], **kwargs):
        import numpy as np

        vectors = np.asarray(self._fn(list(texts)), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    def encode_query(self, query: str, **kwargs):
        return self.encode([query])[0]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.encode(["dimension probe"]).shape[1])
        return self._dimension


class FunctionRetriever(_FunctionComponent, BaseRetriever):
    """A retriever from ``f(query, top_k)`` returning texts, pairs or documents."""

    def retrieve(self, query: str, collection: str, top_k: int = 5, **kwargs):
        out: List[ScoredDocument] = []
        for rank, item in enumerate(self._fn(query, top_k) or []):
            if isinstance(item, ScoredDocument):
                out.append(item)
            elif isinstance(item, tuple) and len(item) == 2:
                out.append(ScoredDocument(content=str(item[0]), score=float(item[1])))
            else:
                out.append(ScoredDocument(content=str(item), score=1.0 / (rank + 1)))
        return out[:top_k]


class FunctionDocumentProcessor(_FunctionComponent, BaseDocumentProcessor):
    """A document processor from ``f(file_path) -> str``."""

    def extract(self, file_path: str, **kwargs) -> ProcessedDocument:
        out = self._fn(file_path)
        return out if isinstance(out, ProcessedDocument) else ProcessedDocument(text=str(out))


_WRAPPERS = {
    "reranker": FunctionReranker,
    "query_enhancer": FunctionQueryEnhancer,
    "chunker": FunctionChunker,
    "embedder": FunctionEmbedder,
    "retriever": FunctionRetriever,
    "document_processor": FunctionDocumentProcessor,
}


def as_component_class(stage: str, obj: Any, name: str) -> type:
    """The class the registry should hold for ``obj`` under ``stage``.

    A class implementing the stage's interface is used as is. A plain function
    is wrapped. Anything else is refused, with the two ways to fix it.
    """
    from .core.registry import _get_expected_abc

    abc = _STAGE_ABCS.get(stage) or _get_expected_abc(stage)
    if abc is None:
        raise ValueError(f"{stage!r} is not a pipeline stage")
    if inspect.isclass(obj):
        if issubclass(obj, abc):
            return obj
        fix = f", or register a plain function with @lasagna.{stage}" if stage in _WRAPPERS else ""
        raise TypeError(
            f"{obj.__qualname__} does not implement the {stage} interface "
            f"({abc.__name__}). Subclass {abc.__name__}{fix}."
        )
    if stage not in _WRAPPERS:
        raise TypeError(
            f"A {stage} is a class, not a function: subclass {abc.__name__} and register the class."
        )
    if not callable(obj):
        raise TypeError(f"{obj!r} is neither a function nor a {abc.__name__} subclass")
    wrapper = _WRAPPERS[stage]
    attrs: Dict[str, Any] = {
        "_fn": staticmethod(obj),
        "_registered_name": name,
        "__doc__": inspect.getdoc(obj) or f"{stage} from {_identity(obj)}",
        "__module__": getattr(obj, "__module__", __name__),
    }
    if stage == "reranker":
        attrs["_listwise"] = _is_listwise(obj)  # decided once, and refused here if unclear
    return type(name.replace(":", "_").replace(".", "_"), (wrapper,), attrs)


def register(
    stage: str,
    obj: Any,
    name: Optional[str] = None,
    *,
    paper: Optional[str] = None,
    claims: Optional[dict] = None,
    version: Optional[str] = None,
) -> Any:
    """Register a function or class as a ``stage`` component; returns ``obj``.

    Built-in names are refused: a function registered as ``cross_encoder``
    would replace the control arm of every sweep in the process, under the
    built-in's name and provenance. Re-registering one of your own names
    replaces it, so a notebook cell can be run again.
    """
    from .core.registry import registry

    name = name or getattr(obj, "__name__", None) or repr(obj)
    if registry.is_builtin(stage, name):
        raise ValueError(
            f"{name!r} is a built-in {stage}; register yours under another name "
            f"(the built-ins are listed by `lasagna component list`)."
        )
    cls = as_component_class(stage, obj, name)
    registry.register_class(
        stage,
        name,
        cls,
        description=cls.__doc__ or "",
        override=True,
        paper=paper,
        claims=claims,
        version=version,
    )
    return obj


def _stage_decorator(stage: str):
    def decorator_or_register(
        name_or_fn: Union[str, Callable, None] = None,
        fn: Optional[Callable] = None,
        *,
        paper: Optional[str] = None,
        claims: Optional[dict] = None,
        version: Optional[str] = None,
    ):
        meta = {"paper": paper, "claims": claims, "version": version}
        if callable(name_or_fn) and not isinstance(name_or_fn, str):
            return register(stage, name_or_fn, **meta)  # @lasagna.reranker
        if fn is not None:
            return register(stage, fn, name_or_fn, **meta)  # lasagna.reranker("n", f)

        def decorate(target):
            return register(stage, target, name_or_fn, **meta)  # @lasagna.reranker("n")

        return decorate

    decorator_or_register.__name__ = stage
    decorator_or_register.__doc__ = (
        f"Register a {stage}: ``@lasagna.{stage}``, ``@lasagna.{stage}('name')`` "
        f"or ``lasagna.{stage}('name', function)``."
    )
    return decorator_or_register


reranker = _stage_decorator("reranker")
query_enhancer = _stage_decorator("query_enhancer")
chunker = _stage_decorator("chunker")
embedder = _stage_decorator("embedder")
retriever = _stage_decorator("retriever")
document_processor = _stage_decorator("document_processor")


def is_reference(value: Any) -> bool:
    """Whether a component name is an import reference such as ``mod:func``."""
    return isinstance(value, str) and ":" in value and not value.startswith(":")


#: The file each loaded reference came from, so a reference never silently
#: means one file in one sweep and another file in the next.
_LOADED_FROM: Dict[Tuple[str, str], str] = {}


def _module_file(module_part: str, search_paths: Sequence[Union[str, Path]]) -> Optional[str]:
    """The file ``module_part`` imports from, looked up the way ``load`` imports it."""
    if module_part.endswith(".py") or "/" in module_part:
        path = Path(module_part)
        if not path.is_absolute():
            for base in (*search_paths, Path.cwd()):
                if (Path(base) / path).exists():
                    return str((Path(base) / path).resolve())
        return str(path.resolve()) if path.exists() else None
    head, *rest = module_part.split(".")
    for base in (*search_paths, Path.cwd()):
        root = Path(base) / head
        target = root.joinpath(*rest) if rest else root
        for candidate in (target.with_suffix(".py"), target / "__init__.py"):
            if candidate.exists():
                return str(candidate.resolve())
    loaded = sys.modules.get(module_part)
    found = getattr(loaded, "__file__", None)
    if found:
        return str(Path(found).resolve())
    try:
        spec = importlib.util.find_spec(module_part)
    except (ImportError, ValueError):
        return None
    return str(Path(spec.origin).resolve()) if spec and spec.origin else None


def load(stage: str, reference: str, search_paths: Sequence[Union[str, Path]] = ()) -> str:
    """Import ``module:attr`` (or ``path/to/file.py:attr``) and register it.

    The reference itself becomes the registered name, so a sweep's arm reads
    ``reranker=my_ideas:keyword_overlap``. ``search_paths`` (the sweep file's
    folder, the working directory) are tried for the module first.

    Only for configuration the user wrote: never resolve references that arrive
    from a network caller, since importing is running code.
    """
    from .core.registry import registry

    module_part, _, attr = reference.rpartition(":")
    if not module_part or not attr:
        raise ValueError(f"{reference!r} is not a 'module:function' reference")
    source = _module_file(module_part, search_paths)
    known = _LOADED_FROM.get((stage, reference))
    if registry.has(stage, reference):
        if known is None:
            return reference  # registered under this name by hand, not loaded here
        if source is not None and source != known:
            raise ImportError(
                f"{reference!r} was already loaded from {known}, and now resolves to "
                f"{source}. Rename one module, or name the file by its path."
            )
        return reference

    if module_part.endswith(".py") or "/" in module_part:
        if source is None:
            raise ImportError(f"cannot load {reference!r}: {module_part} not found")
        path = Path(source)
        # A private module name: loading `utils.py` from a sweep folder must not
        # become the `utils` every later `import utils` in the process gets.
        unique = f"_lasagna_ref_{path.stem}_{hashlib.sha256(source.encode()).hexdigest()[:8]}"
        spec = importlib.util.spec_from_file_location(unique, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {reference!r}: {path} is not a Python file")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique] = module  # dataclasses in the file look themselves up here
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(unique, None)
            raise
    else:
        loaded = sys.modules.get(module_part)
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file and source and str(Path(loaded_file).resolve()) != source:
            raise ImportError(
                f"{reference!r}: the module {module_part!r} is already imported from "
                f"{loaded_file}, not {source}. Rename one module, or name the file by "
                f"its path."
            )
        added = [str(p) for p in (*search_paths, Path.cwd()) if str(p) not in sys.path]
        sys.path[:0] = added
        try:
            module = importlib.import_module(module_part)
        finally:
            for p in added:
                if p in sys.path:
                    sys.path.remove(p)

    target: Any = module
    for part in attr.split("."):
        target = getattr(target, part)
    register(stage, target, reference)
    _LOADED_FROM[(stage, reference)] = source or ""
    return reference


def load_all(stage_values: Iterable[tuple], search_paths: Sequence[Union[str, Path]] = ()) -> None:
    """``load`` every ``(stage, value)`` whose value is an import reference."""
    for stage, value in stage_values:
        if is_reference(value) and (stage in _STAGE_ABCS or stage == "agentic_pipeline"):
            load(stage, value, search_paths)
