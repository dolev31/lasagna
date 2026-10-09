"""Optional adapters that connect Lasagna to third-party agent frameworks.

Nothing here is imported by the rest of the package.  Each module pulls its
framework in lazily, inside the functions that need it, so installing
Lasagna never drags in an agent framework you do not use.

Available adapters:

- :mod:`lasagna.integrations.cuga_agent` -- evaluate a full CUGA agent
  end-to-end instead of the RAG pipeline alone (``pip install cuga``).

To add your own, copy ``cuga_agent.py``: keep the framework import inside
the function body and raise ``ImportError`` with an install hint when the
dependency is missing.
"""
