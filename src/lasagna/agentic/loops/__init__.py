"""Published agentic retrieval loops, each a ``BaseAgenticPipeline``.

Registered in the core registry under the ``agentic_pipeline`` stage:

* ``self_ask`` -- Self-Ask (Press et al. 2022): follow-up questions, each
  answered from retrieval, until the model states its final answer.

Every loop searches through the pipeline's own search (see ``_base``), so an
agentic arm measures the loop inside the complete pipeline.
"""
