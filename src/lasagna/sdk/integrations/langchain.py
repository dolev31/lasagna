"""LangChain Integration.

Provides LangChain-compatible retriever and document loader.

Example:
    ```python
    from lasagna.sdk import KnowledgeClient
    from lasagna.sdk.integrations import KnowledgeRetriever

    client = KnowledgeClient()
    retriever = KnowledgeRetriever(client)

    # Use with LangChain
    docs = await retriever.ainvoke("What is the revenue?")

    # Use in chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    ```
"""

from typing import List, Optional, Any, TYPE_CHECKING
import asyncio
import concurrent.futures

if TYPE_CHECKING:
    from ..client import KnowledgeClient


def _run_async_in_sync(coro):
    """Run an async coroutine from a sync context, handling nested event loops.

    This function properly handles the case where we're already inside an
    async context (running event loop) by using a thread pool to run the
    coroutine in a separate thread with its own event loop.

    Args:
        coro: Async coroutine to run

    Returns:
        Result of the coroutine
    """
    try:
        # Check if we're in an async context
        _loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)

    # Already in async context - run in thread pool with new event loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document as LCDocument
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.document_loaders import BaseLoader

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseRetriever = object
    BaseLoader = object
    LCDocument = None
    CallbackManagerForRetrieverRun = None


class KnowledgeRetriever(BaseRetriever):
    """LangChain retriever backed by Knowledge SDK.

    Implements the LangChain BaseRetriever interface, so chains and agents
    can use it wherever they take a retriever.

    Attributes:
        client: KnowledgeClient instance
        top_k: Number of results to return
        min_score: Minimum similarity score threshold

    Example:
        ```python
        from lasagna.sdk import KnowledgeClient
        from lasagna.sdk.integrations import KnowledgeRetriever

        client = KnowledgeClient()
        retriever = KnowledgeRetriever(client, top_k=5)

        # Async retrieval
        docs = await retriever.ainvoke("What is the revenue?")
        for doc in docs:
            print(doc.page_content[:100])

        # Use in RAG chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        prompt = ChatPromptTemplate.from_template(
            "Answer based on: {context}\\n\\nQuestion: {question}"
        )
        llm = ChatOpenAI()

        chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt
            | llm
        )
        answer = await chain.ainvoke("What is the revenue?")
        ```
    """

    # Pydantic model fields
    client: Any  # KnowledgeClient
    top_k: int = 5
    min_score: float = 0.0
    document_ids: Optional[List[str]] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        client: "KnowledgeClient",
        top_k: int = 5,
        min_score: float = 0.0,
        document_ids: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize retriever.

        Args:
            client: KnowledgeClient instance
            top_k: Number of results to return
            min_score: Minimum similarity score
            document_ids: Optional document filter
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for KnowledgeRetriever. "
                "Install with: pip install langchain-core"
            )

        super().__init__(
            client=client,
            top_k=top_k,
            min_score=min_score,
            document_ids=document_ids,
            **kwargs,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[LCDocument]:
        """Synchronous retrieval with proper event loop handling.

        Handles the case where this is called from an existing async context
        by running the async method in a separate thread with its own event loop.
        """
        return _run_async_in_sync(self._aget_relevant_documents(query, run_manager=run_manager))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[LCDocument]:
        """Async retrieval of relevant documents.

        Args:
            query: Search query
            run_manager: LangChain callback manager

        Returns:
            List of LangChain Document objects
        """
        # Get effective document IDs
        doc_ids = self.document_ids or self.client.focused_document_ids

        # Search
        results = await self.client.search_documents(
            query=query,
            top_k=self.top_k,
            min_score=self.min_score,
            document_ids=doc_ids,
        )

        # Convert to LangChain documents
        documents = []
        for result in results:
            metadata = {
                "document_id": result.document.document_id,
                "filename": result.document.filename,
                "chunk_id": result.chunk.chunk_id,
                "chunk_index": result.chunk.chunk_index,
                "score": result.score,
                "source": result.document.filename,
            }

            if result.chunk.page_number:
                metadata["page_number"] = result.chunk.page_number

            if result.citation_id:
                metadata["citation_id"] = result.citation_id

            documents.append(
                LCDocument(
                    page_content=result.chunk.content,
                    metadata=metadata,
                )
            )

        return documents


class KnowledgeDocumentLoader(BaseLoader):
    """LangChain document loader for knowledge base documents.

    Loads documents from the knowledge base as LangChain Documents.

    Example:
        ```python
        from lasagna.sdk.integrations import KnowledgeDocumentLoader

        loader = KnowledgeDocumentLoader(client)
        docs = await loader.aload()

        # Load specific document
        loader = KnowledgeDocumentLoader(client, document_ids=["doc_123"])
        docs = await loader.aload()
        ```
    """

    def __init__(
        self,
        client: "KnowledgeClient",
        document_ids: Optional[List[str]] = None,
        include_chunks: bool = True,
    ):
        """Initialize loader.

        Args:
            client: KnowledgeClient instance
            document_ids: Optional filter by document IDs
            include_chunks: Load individual chunks vs full documents
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for KnowledgeDocumentLoader. "
                "Install with: pip install langchain-core"
            )

        self.client = client
        self.document_ids = document_ids
        self.include_chunks = include_chunks

    def load(self) -> List[LCDocument]:
        """Synchronous load with proper event loop handling.

        Handles the case where this is called from an existing async context
        by running the async method in a separate thread with its own event loop.
        """
        return _run_async_in_sync(self.aload())

    async def aload(self) -> List[LCDocument]:
        """Async load documents from knowledge base.

        Returns:
            List of LangChain Document objects
        """
        # Get documents
        all_docs = await self.client.list_documents()

        # Filter if specified
        if self.document_ids:
            all_docs = [d for d in all_docs if d.document_id in self.document_ids]

        documents = []

        for doc in all_docs:
            if self.include_chunks:
                # Load all chunks
                chunks = await self.client.documents.get_chunks(doc.document_id)
                for chunk in chunks:
                    documents.append(
                        LCDocument(
                            page_content=chunk.content,
                            metadata={
                                "document_id": doc.document_id,
                                "filename": doc.filename,
                                "chunk_id": chunk.chunk_id,
                                "chunk_index": chunk.chunk_index,
                                "page_number": chunk.page_number,
                                "source": doc.filename,
                            },
                        )
                    )
            else:
                # Create single document with preview
                content = doc.preview_text or f"Document: {doc.filename}"
                documents.append(
                    LCDocument(
                        page_content=content,
                        metadata={
                            "document_id": doc.document_id,
                            "filename": doc.filename,
                            "chunk_count": doc.chunk_count,
                            "source": doc.filename,
                        },
                    )
                )

        return documents
