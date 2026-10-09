"""LLM Judge for evaluating RAG responses.

Uses the provider abstraction for consistent model loading.
"""

from typing import Dict, Any, Optional
from loguru import logger

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from lasagna.providers import get_llm_provider
from .models import JudgeScore, ScoreLevel


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Your task is to evaluate how well an AI agent answered a question using knowledge from documents.

You will be given:
1. The user's question
2. The expected/reference answer
3. The agent's actual response
4. The RAG context retrieved (if any)

Evaluate the agent's response on the following criteria:

**Scoring Scale (1-5):**
- 5 (Excellent): Perfect or near-perfect answer. Fully addresses the question with accurate, complete information.
- 4 (Good): Correct answer with minor omissions or slightly less detail than expected.
- 3 (Partial): Partially correct. Captures some key points but misses important information.
- 2 (Poor): Mostly incorrect or largely irrelevant. May have some tangentially related content.
- 1 (Fail): Completely wrong, no relevant information, or no answer provided.

**Evaluation Criteria:**
- Relevance: Does the answer directly address the question?
- Completeness: Does it cover all key points from the expected answer?
- Factual Accuracy: Is the information accurate based on the context/expected answer?

Be strict but fair. Compare against the expected answer as the ground truth."""


JUDGE_USER_PROMPT = """Evaluate the following:

**User Question:**
{question}

**Expected Answer:**
{expected}

**Agent's Response:**
{response}

**RAG Context Retrieved:**
{context}

Provide your evaluation as a JSON object with these fields:
- score: integer from 1-5
- level: one of "excellent", "good", "partial", "poor", "fail"
- reasoning: brief explanation for your score
- relevance: float from 0.0-1.0
- completeness: float from 0.0-1.0
- factual_accuracy: float from 0.0-1.0"""


class LLMJudge:
    """LLM-as-a-judge evaluator for RAG responses.

    Uses the provider abstraction for model loading, ensuring consistency
    with how models are loaded elsewhere in the codebase.
    """

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        platform: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """Initialize the LLM judge.

        Args:
            model: Pre-instantiated model (optional).
            platform: Model platform if creating new model.
            model_name: Model name (uses platform default if not specified).
            temperature: Temperature for judge responses (default 0 for consistency).
        """
        self._model = model
        self._platform = platform
        self._model_name = model_name
        self._temperature = temperature
        self._initialized = False
        self._init_lock: Optional[Any] = None  # Lazy asyncio.Lock

    async def _ensure_initialized(self) -> None:
        """Lazily initialize the judge model (thread-safe)."""
        if self._initialized:
            return

        # Lazily create lock (avoids requiring asyncio at __init__ time)
        if self._init_lock is None:
            import asyncio

            self._init_lock = asyncio.Lock()

        async with self._init_lock:
            if self._initialized:
                return

            if self._model is None:
                self._model = self._create_judge_model()

            self._prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", JUDGE_SYSTEM_PROMPT),
                    ("human", JUDGE_USER_PROMPT),
                ]
            )

            self._initialized = True
            logger.info(f"LLM Judge initialized with {type(self._model).__name__}")

    def _create_judge_model(self) -> BaseChatModel:
        """Create the judge model using the provider abstraction."""
        model_settings: Dict[str, Any] = {
            "platform": self._platform,
            "temperature": self._temperature,
            "max_tokens": 1000,
        }

        if self._model_name:
            model_settings["model_name"] = self._model_name

        return get_llm_provider().get_chat_model(model_settings)

    async def evaluate(
        self,
        question: str,
        expected: str,
        response: str,
        context: Optional[str] = None,
    ) -> JudgeScore:
        """Evaluate an agent response against expected answer.

        Args:
            question: The user's question.
            expected: The expected/reference answer.
            response: The agent's actual response.
            context: The RAG context retrieved (if any).

        Returns:
            JudgeScore with evaluation results.
        """
        await self._ensure_initialized()

        context_str = context if context else "(No RAG context was retrieved)"

        # Format the prompt
        messages = self._prompt.format_messages(
            question=question,
            expected=expected,
            response=response,
            context=context_str,
        )

        try:
            # Use structured output if available
            if hasattr(self._model, "with_structured_output"):
                structured_model = self._model.with_structured_output(JudgeScore)
                result = await structured_model.ainvoke(messages)
                return result
            else:
                # Fallback to parsing JSON from response
                return await self._evaluate_with_parsing(messages)

        except Exception as e:
            logger.warning(f"Judge evaluation failed: {e}")
            # Return a default low score on failure
            return JudgeScore(
                score=1,
                level=ScoreLevel.FAIL,
                reasoning=f"Evaluation failed: {str(e)}",
                relevance=0.0,
                completeness=0.0,
                factual_accuracy=0.0,
            )

    async def _evaluate_with_parsing(self, messages) -> JudgeScore:
        """Fallback evaluation with manual JSON parsing."""
        import json

        response = await self._model.ainvoke(messages)
        content = response.content

        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            data = json.loads(json_str)

            return JudgeScore(
                score=int(data.get("score", 1)),
                level=ScoreLevel(data.get("level", "fail")),
                reasoning=data.get("reasoning", "No reasoning provided"),
                relevance=float(data.get("relevance", 0.0)),
                completeness=float(data.get("completeness", 0.0)),
                factual_accuracy=float(data.get("factual_accuracy", 0.0)),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse judge response: {e}")
            raise

    async def batch_evaluate(
        self,
        cases: list,
        max_concurrent: int = 3,
    ) -> list:
        """Evaluate multiple cases with concurrency control.

        Args:
            cases: List of (question, expected, response, context) tuples.
            max_concurrent: Maximum concurrent evaluations.

        Returns:
            List of JudgeScore objects.
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def eval_one(case):
            async with semaphore:
                question, expected, response, context = case
                return await self.evaluate(question, expected, response, context)

        return await asyncio.gather(*[eval_one(c) for c in cases])
