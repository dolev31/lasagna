# Python API

These are the names `import lasagna` exposes, in `lasagna.__all__`. The package
loads each one on first use, so `import lasagna` stays fast.

## Decorators and registry

Each stage has a decorator with the same name: `lasagna.reranker`,
`lasagna.query_enhancer`, `lasagna.chunker`, `lasagna.embedder`,
`lasagna.retriever`, `lasagna.document_processor` and `lasagna.generator`. All
take the same arguments and register under the stage they are named for:

```python
import lasagna

@lasagna.reranker                      # name taken from the function
def by_length(query, text): ...

@lasagna.reranker("keyword_overlap")   # explicit name
def keyword_overlap(query, text): ...

lasagna.reranker("by_length_2", by_length)   # a call, not a decorator
```

Each accepts `paper=`, `claims=` and `version=` keywords, which a run records as the
component's provenance. The function each stage expects is listed in
[write a component](../guides/write-a-component.md#a-function-per-stage).

::: lasagna.components.register
    options:
      heading_level: 3

The registry holds every component, built in or yours. `from lasagna.core.registry import registry` gives you the shared instance (also `lasagna.registry`).

::: lasagna.core.registry.ComponentRegistry
    options:
      heading_level: 3
      members:
        - list_components
        - provenance
        - is_builtin
        - has
        - get_class
      show_root_heading: true
      show_root_full_path: false

## LangChain and LlamaIndex

::: lasagna.sdk.integrations.inbound
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      show_docstring_description: false
      members:
        - from_langchain
        - from_llamaindex

## Evaluation

::: lasagna.knowledge_evaluation_sdk.experiment
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      show_docstring_description: false
      members:
        - Experiment
        - ExperimentConfig

## Client and backend

::: lasagna.sdk
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      show_docstring_description: false
      members:
        - KnowledgeClient
        - KnowledgeConfig

::: lasagna.api
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      show_docstring_description: false
      members:
        - RAGBackend
        - get_rag_backend

::: lasagna.standalone
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      show_docstring_description: false
      members:
        - create_rag_backend

## Vector backends

::: lasagna.vector_backends
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      show_docstring_description: false
      members:
        - VectorStoreAdapter
        - register_backend
