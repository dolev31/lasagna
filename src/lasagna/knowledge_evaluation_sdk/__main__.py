"""Allow running as: python -m lasagna.knowledge_evaluation_sdk"""

"""

python -m lasagna.knowledge_evaluation_sdk \
    -d "src/lasagna/knowledge_evaluation_sdk/data/knowledge_base.docx" \
    -g "src/lasagna/knowledge_evaluation_sdk/data/ground-truth.jsonl" \
    -o ./results \
    --verbose \
    --full-agent
    

python -m lasagna.knowledge_evaluation_sdk \
    -d "src/lasagna/knowledge_evaluation_sdk/data/playbook.docx" \
    -g "src/lasagna/knowledge_evaluation_sdk/data/playbook_ground_truth.jsonl" \
    -o ./results \
    --verbose


"""
from .cli import main  # noqa: E402

if __name__ == "__main__":
    main()
