"""Test document corpus generation with ground-truth relevance judgments.

Generates documents with domain-specific content across diverse topics
to test cross-topic isolation and provide ground-truth for IR metrics.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TestDocument:
    """A test document with ground-truth metadata."""

    path: Path
    filename: str
    topic: str
    content_summary: str
    unique_marker: str  # unique string only in this document
    ground_truth: Dict[str, int] = field(default_factory=dict)
    # Maps query -> relevance grade (0=irrelevant, 1=marginal, 2=highly relevant)


@dataclass
class TestCorpus:
    """Complete test corpus with ground-truth relevance judgments."""

    documents: List[TestDocument]
    queries: List[GroundTruthQuery]


@dataclass
class GroundTruthQuery:
    """A query with expected relevant documents."""

    query: str
    relevance: Dict[str, int]  # doc unique_marker -> relevance grade
    topic: str


# ---------------------------------------------------------------------------
# Document content templates
# ---------------------------------------------------------------------------

_TOPICS = [
    {
        "topic": "quantum_physics",
        "title": "Introduction to Quantum Mechanics",
        "content": (
            "Quantum mechanics is a fundamental theory in physics that describes nature "
            "at the smallest scales of energy levels of atoms and subatomic particles. "
            "The wave function, denoted by the Greek letter psi, provides a complete "
            "description of a quantum system. The Heisenberg uncertainty principle states "
            "that the position and momentum of a particle cannot both be precisely "
            "determined at the same time. Quantum entanglement is a phenomenon where "
            "pairs of particles interact in ways such that the quantum state of each "
            "particle cannot be described independently. Schrodinger's equation governs "
            "how the quantum state evolves over time. Quantum superposition allows "
            "particles to exist in multiple states simultaneously until measured. "
            "The Copenhagen interpretation suggests that quantum mechanics does not "
            "yield a description of an objective reality but deals only with probabilities."
        ),
        "queries": [
            ("What is the Heisenberg uncertainty principle?", 2),
            ("How does quantum entanglement work?", 2),
            ("What is Schrodinger's equation?", 2),
        ],
    },
    {
        "topic": "machine_learning",
        "title": "Deep Learning Fundamentals",
        "content": (
            "Deep learning is a subset of machine learning that uses neural networks "
            "with multiple layers to model complex patterns in data. Convolutional "
            "neural networks (CNNs) are particularly effective for image recognition "
            "tasks, using convolutional filters to detect spatial features. Recurrent "
            "neural networks (RNNs) and their variants like LSTM and GRU are designed "
            "for sequential data processing. The transformer architecture, introduced "
            "in the 'Attention Is All You Need' paper, revolutionized natural language "
            "processing through self-attention mechanisms. Backpropagation is the "
            "algorithm used to compute gradients for training neural networks. "
            "Regularization techniques like dropout, batch normalization, and weight "
            "decay help prevent overfitting. Transfer learning allows pre-trained "
            "models to be fine-tuned for specific tasks with limited data."
        ),
        "queries": [
            ("How do convolutional neural networks work?", 2),
            ("What is the transformer architecture?", 2),
            ("What is backpropagation in neural networks?", 2),
        ],
    },
    {
        "topic": "climate_science",
        "title": "Climate Change and Global Warming",
        "content": (
            "Climate change refers to significant changes in global temperatures and "
            "weather patterns over time. The greenhouse effect occurs when gases in "
            "Earth's atmosphere trap heat from the sun. Carbon dioxide, methane, and "
            "nitrous oxide are the primary greenhouse gases responsible for global "
            "warming. The Intergovernmental Panel on Climate Change (IPCC) provides "
            "scientific assessments on climate change. Sea level rise is caused by "
            "thermal expansion of water and melting of ice sheets and glaciers. "
            "The Paris Agreement aims to limit global temperature increase to 1.5 "
            "degrees Celsius above pre-industrial levels. Ocean acidification, caused "
            "by absorption of excess CO2, threatens marine ecosystems and coral reefs. "
            "Renewable energy sources like solar, wind, and hydroelectric power are "
            "essential for reducing greenhouse gas emissions."
        ),
        "queries": [
            ("What causes the greenhouse effect?", 2),
            ("What is the Paris Agreement on climate change?", 2),
            ("How does ocean acidification affect marine life?", 2),
        ],
    },
    {
        "topic": "ancient_history",
        "title": "The Roman Empire: Rise and Fall",
        "content": (
            "The Roman Empire, at its peak under Emperor Trajan in 117 AD, encompassed "
            "much of Europe, North Africa, and the Middle East. The Roman Republic "
            "transitioned to an empire under Augustus Caesar in 27 BC. Roman engineering "
            "achievements include aqueducts, roads, and the Colosseum. The Pax Romana, "
            "a period of relative peace lasting approximately 200 years, allowed trade "
            "and culture to flourish. The Roman legal system established principles "
            "that form the basis of many modern legal systems. The fall of the Western "
            "Roman Empire in 476 AD was caused by a combination of military pressure "
            "from barbarian tribes, economic instability, political corruption, and "
            "overexpansion. The Eastern Roman Empire, or Byzantine Empire, continued "
            "for nearly another thousand years until 1453 AD."
        ),
        "queries": [
            ("When did the Roman Empire reach its greatest extent?", 2),
            ("What was the Pax Romana?", 2),
            ("What caused the fall of the Roman Empire?", 2),
        ],
    },
    {
        "topic": "molecular_biology",
        "title": "DNA Structure and Gene Expression",
        "content": (
            "Deoxyribonucleic acid (DNA) is a molecule that carries genetic instructions "
            "for the development and functioning of all known organisms. The double "
            "helix structure of DNA was discovered by Watson and Crick in 1953. DNA "
            "replication is the process by which DNA makes a copy of itself during "
            "cell division. Transcription converts DNA sequences into messenger RNA "
            "(mRNA), which is then translated into proteins by ribosomes. CRISPR-Cas9 "
            "is a revolutionary gene editing technology that allows precise modifications "
            "to DNA sequences. Epigenetic modifications, such as DNA methylation and "
            "histone modification, regulate gene expression without changing the DNA "
            "sequence itself. The Human Genome Project, completed in 2003, mapped all "
            "human genes and has been instrumental in advancing personalized medicine."
        ),
        "queries": [
            ("Who discovered the structure of DNA?", 2),
            ("How does CRISPR-Cas9 gene editing work?", 2),
            ("What are epigenetic modifications?", 2),
        ],
    },
    {
        "topic": "cybersecurity",
        "title": "Network Security Fundamentals",
        "content": (
            "Network security involves protecting computer networks from unauthorized "
            "access, misuse, and attacks. Firewalls act as barriers between trusted "
            "internal networks and untrusted external networks. Public key infrastructure "
            "(PKI) uses asymmetric cryptography with public and private key pairs for "
            "secure communication. SQL injection attacks exploit vulnerabilities in "
            "database queries to access or manipulate data. Cross-site scripting (XSS) "
            "allows attackers to inject malicious scripts into web pages viewed by other "
            "users. Multi-factor authentication (MFA) requires users to provide two or "
            "more verification factors. Zero-trust architecture assumes no user or system "
            "should be trusted by default, requiring verification for every access "
            "request. Intrusion detection systems (IDS) monitor network traffic for "
            "suspicious activity and potential threats."
        ),
        "queries": [
            ("What is a SQL injection attack?", 2),
            ("How does public key infrastructure work?", 2),
            ("What is zero-trust architecture?", 2),
        ],
    },
    {
        "topic": "economics",
        "title": "Monetary Policy and Central Banking",
        "content": (
            "Monetary policy is the process by which central banks manage the money "
            "supply and interest rates to influence economic activity. The Federal "
            "Reserve, established in 1913, is the central banking system of the United "
            "States. Quantitative easing is an unconventional monetary policy where "
            "central banks purchase government securities to increase the money supply. "
            "Inflation targeting is a strategy where central banks set an explicit "
            "target for the inflation rate. The Phillips curve describes the inverse "
            "relationship between inflation and unemployment. Interest rate decisions "
            "affect borrowing costs, consumer spending, and investment. The Taylor rule "
            "provides a formula for setting interest rates based on inflation and output "
            "gaps. Cryptocurrency and digital currencies present new challenges for "
            "monetary policy and financial regulation."
        ),
        "queries": [
            ("What is quantitative easing?", 2),
            ("How does the Phillips curve relate inflation and unemployment?", 2),
            ("What is the Taylor rule in monetary policy?", 2),
        ],
    },
    {
        "topic": "space_exploration",
        "title": "Mars Exploration and Colonization",
        "content": (
            "Mars exploration has been a focus of space agencies worldwide, with "
            "missions like NASA's Perseverance rover and ESA's ExoMars program. "
            "The Martian atmosphere is composed primarily of carbon dioxide with "
            "a surface pressure about 1% of Earth's. Olympus Mons, the tallest "
            "volcano in the solar system, rises 21.9 kilometers above the Martian "
            "surface. Evidence of ancient water flow on Mars suggests the planet "
            "may have once supported microbial life. In-situ resource utilization "
            "(ISRU) technology aims to produce oxygen, water, and fuel from Martian "
            "resources. SpaceX's Starship is being developed as a fully reusable "
            "spacecraft for Mars colonization. Radiation exposure during the 6-9 "
            "month transit and on the Martian surface presents significant health "
            "risks for future colonists."
        ),
        "queries": [
            ("What is the composition of Mars atmosphere?", 2),
            ("What is Olympus Mons?", 2),
            ("What are the health risks of Mars colonization?", 2),
        ],
    },
]


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------


def generate_test_corpus(
    num_docs: int = 20,
    output_dir: Optional[Path] = None,
) -> TestCorpus:
    """Generate a test corpus with ground-truth relevance judgments.

    Args:
        num_docs: Maximum number of documents to generate (capped at available topics).
        output_dir: Directory to write files to. Uses tempdir if None.

    Returns:
        TestCorpus with documents and ground-truth queries.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="stress_corpus_"))

    output_dir.mkdir(parents=True, exist_ok=True)

    documents: List[TestDocument] = []
    all_queries: List[GroundTruthQuery] = []

    topics_to_use = _TOPICS[: min(num_docs, len(_TOPICS))]

    for topic_data in topics_to_use:
        marker = f"UNIQUE-{topic_data['topic'].upper()}-{uuid.uuid4().hex[:8]}"

        # Write document file
        filename = f"{topic_data['topic']}.txt"
        content = (
            f"# {topic_data['title']}\n\n{topic_data['content']}\n\nDocument identifier: {marker}\n"
        )
        file_path = output_dir / filename
        file_path.write_text(content)

        doc = TestDocument(
            path=file_path,
            filename=filename,
            topic=topic_data["topic"],
            content_summary=topic_data["title"],
            unique_marker=marker,
        )
        documents.append(doc)

        # Build ground-truth queries
        for query_text, relevance in topic_data["queries"]:
            # This query is highly relevant to this document
            all_queries.append(
                GroundTruthQuery(
                    query=query_text,
                    relevance={marker: relevance},
                    topic=topic_data["topic"],
                )
            )

    return TestCorpus(documents=documents, queries=all_queries)


def generate_tenant_unique_document(
    tenant_id: str,
    output_dir: Optional[Path] = None,
) -> TestDocument:
    """Generate a document with content unique to a specific tenant.

    Used for tenant isolation tests — content includes a unique marker
    that should only be findable by the owning tenant.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="stress_tenant_"))

    output_dir.mkdir(parents=True, exist_ok=True)

    marker = f"TENANT-SECRET-{tenant_id}-{uuid.uuid4().hex[:12]}"
    filename = f"tenant_doc_{tenant_id[:8]}.txt"
    content = (
        f"Confidential Document for Tenant {tenant_id}\n\n"
        f"This document contains proprietary information: {marker}\n\n"
        f"The secret identification code for this tenant is {marker}. "
        f"This code should only be accessible to authorized personnel "
        f"within the tenant organization identified as {tenant_id}.\n"
    )

    file_path = output_dir / filename
    file_path.write_text(content)

    return TestDocument(
        path=file_path,
        filename=filename,
        topic="tenant_isolation",
        content_summary=f"Tenant-specific document for {tenant_id[:8]}",
        unique_marker=marker,
    )
