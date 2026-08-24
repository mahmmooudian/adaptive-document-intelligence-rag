# Adaptive Document Intelligence RAG

> An adaptive document intelligence platform exploring multiple Retrieval-Augmented Generation strategies for fast, grounded, and context-aware document analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![RAG](https://img.shields.io/badge/RAG-Adaptive-purple)
![LLM](https://img.shields.io/badge/LLM-GPT--OSS--20B-green)
![Provider](https://img.shields.io/badge/Inference-Groq-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

Retrieval-Augmented Generation systems are often built around a fixed retrieval pipeline.

That approach can work well for simple document question answering, but the same retrieval depth is not always appropriate for every question.

A direct factual query may only need a small amount of highly relevant evidence.

A more analytical question may require information from several sections of a document.

And more complex tasks may require structural understanding, relationships between entities, or multi-step reasoning.

**Adaptive Document Intelligence RAG** explores a different direction:

> Instead of forcing every question through the same RAG pipeline, can the system adapt its retrieval and reasoning strategy to the complexity of the query?

The project currently explores three architectural directions:

- **TurboRAG**
- **Hierarchical Intelligence RAG**
- **Graph-Hybrid Agentic RAG**

The first working implementation is **TurboRAG v3.1**.

---

# Architecture Direction

## 1. TurboRAG

TurboRAG is the fast retrieval layer of the project.

It is designed for efficient document question answering while keeping generated responses grounded in retrieved evidence.

At a high level:

```text
Document
   ↓
Document Processing
   ↓
Page-Aware Chunking
   ↓
Semantic Retrieval + Lexical Retrieval
   ↓
Hybrid Candidate Retrieval
   ↓
Cross-Encoder Reranking
   ↓
Evidence Quality Filtering
   ↓
Evidence Deduplication
   ↓
Adaptive Evidence Selection
   ↓
Grounded Generation
   ↓
Answer + Citations + Metrics
```

The objective is to retrieve enough evidence to answer the question without unnecessarily sending large amounts of document context to the language model.

---

## 2. Hierarchical Intelligence RAG

The second architectural direction focuses on larger and more structured documents.

In many real-world documents, information exists at several levels:

```text
Document
   ↓
Section
   ↓
Subsection
   ↓
Paragraph
   ↓
Evidence
```

Hierarchical Intelligence RAG is intended to preserve more of this structure during retrieval.

Instead of treating every chunk as an independent unit, this architecture explores relationships between local evidence and the broader document context.

Potential applications include:

- Technical reports
- Research papers
- Legal documents
- Policies
- Enterprise documentation
- Large operational manuals

This architecture is currently part of the next development stage.

---

## 3. Graph-Hybrid Agentic RAG

The third architectural direction targets more complex information needs.

Some questions cannot be answered from a single retrieved passage.

They may require:

- Connecting multiple facts
- Following relationships between entities
- Retrieving evidence from multiple locations
- Multi-step retrieval
- Iterative reasoning

Graph-Hybrid Agentic RAG explores the combination of graph-oriented knowledge representation, hybrid retrieval, and agentic reasoning.

This architecture is currently under exploration and has not yet reached the same implementation stage as TurboRAG.

---

# Current Implementation

## TurboRAG v3.1

The current working proof of concept includes:

### Document Processing

- PDF support
- DOCX support
- TXT support
- Page-aware PDF extraction
- Text normalization
- Document cleaning
- Context-aware chunk generation

### Retrieval

- Multilingual semantic embeddings
- Dense vector retrieval
- BM25 lexical retrieval
- Hybrid retrieval
- Cross-encoder reranking

### Evidence Processing

- Evidence quality filtering
- Duplicate evidence reduction
- Query complexity detection
- Adaptive evidence selection
- Citation-aware context construction

### Generation

- Evidence-grounded answer generation
- Groq API inference
- GPT-OSS 20B
- Source-aware prompting
- Explicit insufficient-evidence responses

### Observability

- Retrieval latency
- Reranking latency
- Generation latency
- End-to-end query latency
- Prompt token usage
- Completion token usage
- Evidence-oriented confidence indicator

---

# Adaptive Evidence Selection

One of the main experimental components of TurboRAG is adaptive evidence selection.

A simple factual query does not necessarily require the same amount of context as an analytical question.

For example:

```text
What is the primary contract number?
```

can follow a lightweight path:

```text
Query
 ↓
Focused Retrieval
 ↓
Small Evidence Set
 ↓
Grounded Answer
```

A more analytical query such as:

```text
Why was the project budget increased and what was the final revised budget?
```

may require:

```text
Query
 ↓
Broader Retrieval
 ↓
Reranking
 ↓
Multiple Evidence Sources
 ↓
Grounded Synthesis
```

The current implementation classifies query complexity and adjusts the amount of final evidence accordingly.

---

# Grounded Generation

The language model does not receive the complete document.

Instead, TurboRAG retrieves and ranks relevant evidence before constructing the final context.

The answer generator is instructed to use only the supplied evidence.

Example:

```text
Question:
What is the primary contract number?

Answer:
The primary contract number is CN-AX91. [SOURCE_1]
```

The citation can then be mapped back to document metadata such as:

```text
Document: test_document.pdf
Page: 2
Chunk: 2
```

This provides a simple traceability layer between generated answers and the source document.

---

# Insufficient Evidence Handling

A document intelligence system should not create an answer when the required information does not exist in the available evidence.

For example:

```text
What is the CEO's phone number?
```

If the document contains no supporting evidence, TurboRAG is instructed to return:

```text
The available document evidence is insufficient to answer this question.
```

This behavior is part of the project's effort to reduce unsupported answers.

---

# Example Evaluation

The repository includes a synthetic technical document for controlled RAG testing.

The document contains information distributed across multiple pages and sections, allowing several retrieval behaviors to be tested.

## Test 1 — Direct Fact Retrieval

```text
What is the primary contract number?
```

Expected behavior:

```text
Retrieve direct evidence
        ↓
Generate concise answer
        ↓
Attach source citation
```

---

## Test 2 — Multi-Evidence Question

```text
Why was the project budget increased and what was the final revised budget?
```

Expected behavior:

```text
Retrieve related sections
        ↓
Rerank evidence
        ↓
Select multiple supporting sources
        ↓
Generate grounded synthesis
```

---

## Test 3 — Unsupported Question

```text
What is the CEO's phone number?
```

Expected behavior:

```text
Weak / unrelated evidence
        ↓
Insufficient evidence detected
        ↓
No unsupported answer
```

---

# Current Proof-of-Concept Results

During the current controlled evaluation, the system successfully demonstrated:

```text
Document Parsing              ✓
Page-Aware Processing         ✓
Context-Aware Chunking        ✓
Multilingual Embeddings       ✓
Vector Retrieval              ✓
BM25 Retrieval                ✓
Hybrid Retrieval              ✓
Cross-Encoder Reranking       ✓
Evidence Quality Filtering    ✓
Evidence Deduplication        ✓
Adaptive Evidence Selection   ✓
Query Complexity Detection    ✓
Grounded LLM Generation       ✓
Source Citation               ✓
Insufficient-Evidence Handling✓
Latency Monitoring            ✓
Token Monitoring              ✓
Interactive Streamlit UI      ✓
```

The current results represent proof-of-concept testing rather than a production benchmark.

---

# Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| PDF Processing | PyMuPDF |
| DOCX Processing | python-docx |
| Semantic Embeddings | Sentence Transformers |
| Lexical Retrieval | BM25 |
| Reranking | Cross-Encoder |
| Numerical Processing | NumPy |
| ML Utilities | Scikit-learn |
| LLM Provider | Groq |
| Generation Model | GPT-OSS 20B |
| UI | Streamlit |
| Configuration | python-dotenv |

---

# Repository

Clone the project:

```bash
git clone https://github.com/mahmmooudian/adaptive-document-intelligence-rag.git
```

Enter the project directory:

```bash
cd adaptive-document-intelligence-rag
```

---

# Installation

## 1. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Create a file named:

```text
.env
```

in the project root.

Add:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Do not commit your real API key.

The `.env` file is excluded from version control through `.gitignore`.

---

# Running TurboRAG

## Terminal Version

Run:

```bash
python demo_turborag.py
```

You will be asked for a document path:

```text
Enter document path:
```

Example:

```text
C:\path\to\document.pdf
```

After indexing completes:

```text
Ask a question (or type 'exit'):
```

You can then query the document interactively.

---

# Running the Streamlit Interface

Start the application:

```bash
streamlit run app.py
```

Streamlit will normally open:

```text
http://localhost:8501
```

The interface currently provides:

- Document upload
- Automatic document indexing
- Question answering
- Grounded responses
- Source citations
- Confidence information
- Query classification
- Retrieval metrics
- Evidence inspection
- Token usage
- Response-time monitoring

---

# Streamlit Workflow

```text
Upload Document
       ↓
Process & Index
       ↓
System Ready
       ↓
Ask Question
       ↓
Hybrid Retrieval
       ↓
Reranking
       ↓
Adaptive Evidence
       ↓
Grounded Generation
       ↓
Answer
       ↓
Sources + Metrics
```

---

# Project Structure

The public repository is organized around modular RAG components and supporting resources.

```text
adaptive-document-intelligence-rag/
│
├── core/
│
├── data/
│
├── docs/
│
├── llm/
│
├── models/
│
├── utils/
│
├── app.py
├── demo_turborag.py
├── requirements.txt
├── test_document.pdf
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

Some folders may evolve as the hierarchical and graph-based architectures are implemented.

---

# Design Principles

## Groundedness

Generated answers should be supported by retrieved document evidence.

## Traceability

Users should be able to identify which source contributed to an answer.

## Efficiency

Simple questions should avoid unnecessarily large contexts and expensive reasoning.

## Adaptivity

Retrieval depth should be able to change based on query complexity.

## Modularity

Parsing, retrieval, reranking, evidence selection, and generation should remain independently replaceable.

## Observability

Latency, evidence selection, and token consumption should remain visible during experimentation.

---

# Security

Never commit secrets to the repository.

The following should remain local:

```text
.env
API keys
credentials
virtual environments
temporary uploads
model caches
Python cache files
local development files
```

Before pushing changes, always verify:

```bash
git status
```

and confirm that `.env` is not included.

---

# Roadmap

The next development stages may include:

- Hierarchical document indexing
- Section-aware retrieval
- Document-level summaries
- Multi-document retrieval
- Graph knowledge representation
- Entity and relationship extraction
- Multi-hop retrieval
- Agentic query planning
- Retrieval routing
- Improved evidence quality scoring
- Confidence calibration
- RAG evaluation benchmarks
- Comparative latency testing
- Comparative cost testing
- Persistent vector storage
- Production-oriented indexing
- Authentication and access control
- Improved user interface

---

# Research Direction

The broader research question behind this project is:

> **Can a document intelligence system dynamically choose an appropriate retrieval and reasoning strategy based on query complexity while maintaining efficiency, groundedness, and traceability?**

TurboRAG is the first implemented stage toward exploring that question.

Hierarchical Intelligence RAG and Graph-Hybrid Agentic RAG extend the concept toward increasingly complex document understanding and reasoning scenarios.

---

# Current Status

### TurboRAG

```text
Status: Working Proof of Concept
Version: v3.1
```

### Hierarchical Intelligence RAG

```text
Status: Architecture / Development Stage
```

### Graph-Hybrid Agentic RAG

```text
Status: Architecture / Research Stage
```

---

# Public Architecture Notes

The architectural diagrams and documentation in this repository intentionally provide a high-level representation of the system.

Some experimental parameters, internal configuration choices, prompt strategies, thresholds, evaluation procedures, and implementation-specific design decisions are intentionally omitted.

---

# Disclaimer

This repository is an active research and engineering proof of concept.

It is not currently intended to represent a fully production-hardened document intelligence platform.

Results may change as retrieval strategies, evaluation methods, models, and architectures evolve.

---

# Author

**Amir Mohammad Mahmoudian**

AI / Machine Learning Developer

GitHub: [@mahmmooudian](https://github.com/mahmmooudian)

---

# License

This project is released under the **MIT License**.

See the `LICENSE` file for details.

---

## Project Repository

**Adaptive Document Intelligence RAG**

https://github.com/mahmmooudian/adaptive-document-intelligence-rag
