This folder contains the results files from all architectures in `jsonl` format. It is kept here for further evaluation. There is also a classification folder which was used during testing to see if GEval and BertScore could be used for answer evaluation. As well as results scripts and parsers.
---
* `classification/` folder containing testing og BertScore and GEval.
* `helper_functions` is a shared module of utility functions used across the result scripts, covering version control for output files, plotting (stacked bar charts comparing string match vs LLM-judge classifications, pageid recall graphs), answer evaluation (text normalization, numeric extraction and matching, LLM-as-judge classification with `qwen3:0.6b`, hedging detection), and per-architecture recall counters that check whether the gold updated/outdated chunks were exposed to or selected by each pipeline.
* `exact_match.py` is a script taking the answer for each entry for each architecture and exact matches it with the *gold answer*. 
* `LLM-as-judge.py` does the same as `exact_match.py`, but instead of exact matching answers it calls the LLM for evaluating wether the *predicted answer* resembles *outdated gold answer* or *updated gold answer*.
* `models_to_evaluate.json` contains architectures and setups one would like to evaluate.
* `models.json` contains all architectures:
    ```json
    "RCA": {
        "RCA": {
            "RCA_C": 25,
            }
        }
    ```
    - it is structured following: architecture, exact model (essentially setup), setup(unique for RCA since it is only method with aggregation) and version(what version of the result).
* `page_retrieve.py` class divided into selection_recall and exposure_recall