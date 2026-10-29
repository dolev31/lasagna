"""Published agentic retrieval loops, each a ``BaseAgenticPipeline``.

Registered in the core registry under the ``agentic_pipeline`` stage:

* ``self_ask`` -- Self-Ask (Press et al. 2022): follow-up questions, each
  answered from retrieval, until the model states its final answer.
* ``ircot`` -- IRCoT (Trivedi et al. 2023): one chain-of-thought sentence per
  step, each sentence the next retrieval query, up to a step limit.
* ``flare`` -- FLARE (Jiang et al. 2023): forward-looking generation that
  retrieves when the tentative next sentence is not confident.
* ``adaptive`` -- Adaptive-RAG (Jeong et al. 2024): a complexity classifier
  routes each question to no retrieval, one retrieval or IRCoT.
* ``self_rag`` -- Self-RAG (Asai et al. 2023), prompting variant: retrieval on
  demand and a per-passage critique of relevance, support and usefulness.
* ``search_o1`` -- Search-o1 (Li et al. 2025): a reasoning model writes search
  queries mid-reasoning; Reason-in-Documents reads each search's passages into
  the reasoning.

Every loop searches through the pipeline's own search (see ``_base``), so an
agentic arm measures the loop inside the complete pipeline.
"""
