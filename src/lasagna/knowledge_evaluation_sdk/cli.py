"""CLI entry point for Knowledge Evaluation SDK.

Usage:
    python -m lasagna.knowledge_evaluation_sdk.cli \
        --document ./data/knowledge_base.docx \
        --ground-truth ./data/ground_truth.jsonl \
        --output ./results

Or programmatically:
    from lasagna.knowledge_evaluation_sdk import (
        KnowledgeEvaluator,
        EvalConfig,
    )

    evaluator = KnowledgeEvaluator(EvalConfig(...))
    metrics = await evaluator.run()
"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from .models import EvalConfig
from .evaluator import KnowledgeEvaluator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SDK Knowledge/RAG feature against benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python -m lasagna.knowledge_evaluation_sdk.cli \\
      -d ./data/knowledge_base.docx \\
      -g ./data/ground_truth.jsonl

  # With custom output and verbose logging
  python -m lasagna.knowledge_evaluation_sdk.cli \\
      -d ./data/knowledge_base.docx \\
      -g ./data/ground_truth.jsonl \\
      -o ./my_results \\
      --verbose

  # Use specific judge model
  python -m lasagna.knowledge_evaluation_sdk.cli \\
      -d ./data/knowledge_base.docx \\
      -g ./data/ground_truth.jsonl \\
      --judge-platform openai \\
      --judge-model gpt-4o
        """,
    )

    # Required arguments
    parser.add_argument(
        "-d",
        "--document",
        type=str,
        required=True,
        help="Path to knowledge document (PDF, DOCX, TXT, etc.)",
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        type=str,
        required=True,
        help="Path to JSONL ground truth file",
    )

    # Optional arguments
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./eval_results",
        help="Output directory for results (default: ./eval_results)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable reranking in RAG search",
    )
    parser.add_argument(
        "--use-hyde",
        action="store_true",
        help="Enable HyDE query expansion",
    )
    parser.add_argument(
        "--judge-platform",
        type=str,
        default="openai",
        choices=["openai", "azure", "groq", "watsonx", "google-genai", "openrouter"],
        help="Platform for LLM judge (default: openai)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for judge (uses platform default if not specified)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent evaluations (default: 3)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--full-agent",
        action="store_true",
        help="Use full SDK agent invoke flow (default: RAG-only mode)",
    )

    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    )


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    # Validate paths
    doc_path = Path(args.document)
    if not doc_path.exists():
        logger.error(f"Document not found: {doc_path}")
        return 1

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        logger.error(f"Ground truth file not found: {gt_path}")
        return 1

    # Create configuration
    config = EvalConfig(
        document_path=str(doc_path.absolute()),
        ground_truth_path=str(gt_path.absolute()),
        output_dir=args.output,
        top_k=args.top_k,
        use_reranking=not args.no_reranking,
        use_hyde=args.use_hyde,
        judge_model_platform=args.judge_platform,
        judge_model_name=args.judge_model,
        max_concurrent=args.max_concurrent,
        verbose=args.verbose,
    )

    logger.info("=" * 60)
    logger.info("KNOWLEDGE EVALUATION SDK")
    logger.info("=" * 60)
    logger.info(f"Document:     {doc_path.name}")
    logger.info(f"Ground Truth: {gt_path.name}")
    logger.info(f"Output:       {args.output}")
    logger.info(f"Judge:        {args.judge_platform}/{args.judge_model or 'default'}")
    logger.info(f"Mode:         {'Full Agent' if args.full_agent else 'RAG-only'}")
    logger.info("=" * 60)

    # Run evaluation
    evaluator = KnowledgeEvaluator(config, use_full_agent=args.full_agent)

    try:
        metrics = await evaluator.run()

        # Print summary
        print("\n")
        print(metrics.summary())

        # Return exit code based on pass rate
        if metrics.pass_rate >= 80:
            logger.info("Evaluation PASSED (pass rate >= 80%)")
            return 0
        else:
            logger.warning(f"Evaluation FAILED (pass rate {metrics.pass_rate:.1f}% < 80%)")
            return 1

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 2
    finally:
        await evaluator.cleanup()


def main() -> None:
    """CLI entry point."""
    from loguru import logger

    logger.enable("lasagna")
    args = parse_args()
    setup_logging(args.verbose)

    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
