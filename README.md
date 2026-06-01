[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# MSc-Multi-Agent-RAG

Multi-agent Retrieval-Augmented Generation for temporal knowledge disambiguation. MSc thesis project, Data Science, University of Stavanger.

## About

This project investigates whether RAG systems can distinguish between updated and outdated versions of the same Wikipedia article **without access to date metadata**, relying only on linguistic recency cues in the content itself (e.g. *"currently"*, *"as of"*, *"was succeeded by"*). The retrieval layer uses BM25 over a single unified Elasticsearch index of ~1.68M chunks spanning seven monthly Wikipedia snapshots (June–December 2024), so no date information leaks from the index to the agents.

Evaluation is performed on a 500-question sample from the [HoH benchmark](https://arxiv.org/abs/2503.04800) (Ouyang et al., 2025), where each question has both an updated and an outdated gold answer.

## Architectures

Several multi-agent architectures are implemented and compared against a single-agent baseline:

- **Baseline RAG** — pointwise (k=1) and listwise (k=5) variants
- **RTA** (Relevance, Temporal, Aggregation) - weighted aggregation of relevance and recency scores
- **RCA** (Relevance, Candidate, Aggregation) - parallel candidate agents, selection by majority vote, confidence, or random
- **RCO** (Relevance, Conflict resolution, Output) - conflict detection followed by conditional verification
- **RCDS** (Relevance, Candidate, Debate, Supervisor) - multi-round debate between candidate agents, supervised final selection

## Repository structure

- `data/`: HoH dataset, 500-question evaluation set, preprocessing scripts and Elasticsearch indexing
- `WikiAPI/`: scripts for downloading historical Wikipedia snapshots via the MediaWiki API
- `RAG/`: implementation of the baseline and multi-agent architectures
- `results/`: per-architecture result files (`.jsonl`), evaluation utilities, and classification helpers

## How to run

### Prerequisites

- Python 3.10+
- Elasticsearch 8.17 running locally on `localhost:9200`
- Access to an Ollama / OpenWebUI endpoint serving `llama3.3:70b` and `qwen3:0.6b`
- The HoH dataset parquet file placed in `data/clean_HoH_dataset/`

### Setup

```bash
git clone https://github.com/<user>/MSc-Multi-Agent-RAG.git
cd MSc-Multi-Agent-RAG

conda create -n master python=3.10
conda activate master
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API key for the LLM endpoint:<br>
`OLLAMA_API=<your-api-key>`

### Pipeline

The full pipeline runs in four stages. Each stage assumes the previous one has completed.

**1. Extract document IDs and timestamps from the HoH dataset**

```bash
cd data/data_scripts
python extract_ids.py
```

Produces `data/doc_times.json`, a mapping from `pageid` to the list of snapshot dates needed.

**2. Download Wikipedia snapshots**

```bash
cd WikiAPI
python batch_create_snapshots.py
```

Downloads all required revisions into `data/KB_raw/YYYY-MM-DD/`. Long-running; rate-limited to be polite to the MediaWiki API. Failed downloads are logged to `data/failed_snapshots.jsonl`.

**3. Preprocess and index**

```bash
cd data/data_scripts
python data_preprocessing.py       # cleans wikitext -> data/KB_cleaned/
python elasticsearch_indexing.py   # chunks and indexes into 'wikipedia_snapshots'
```

**4. Sample the 500 evaluation questions**

```bash
python Create_500_Q.py
```

Writes `data/500Q/500_hoh_questions.csv` using fixed random seed 42.

### Running the RAG architectures

Each architecture is a standalone script under `RAG/`:

```bash
# Baseline
cd RAG/RAG_baseline && python baseline.py

# Multi-agent
cd RAG/MA-RAG && python RTA.py
cd RAG/MA-RAG && python RCA.py
# ... etc
```

Each script writes a versioned `.jsonl` result file to `results/`.

To run all architectures sequentially in a detached `tmux` session (useful for remote servers)

### Evaluation

Result files are evaluated using string matching followed by LLM-as-judge as a fallback. See `results/` for the evaluation scripts and `helper_functions.py` for the shared utilities.

## Notes
This repository is developed in accordance with the IAI guidelines[^1] and the IAI Python style guide[^2].

27.feb -> create statistics for edge cases (several outdated, several new, etc)
KB folder structure by month



---
[^1]: https://github.com/iai-group/guidelines
[^2]: https://github.com/iai-group/guidelines/tree/main/python
