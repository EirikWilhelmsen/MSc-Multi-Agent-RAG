The content of this folder covers all the architectures, their respective setups, helper functions and agents. The structure is designed to in best manner remove redundant code.
---
* `MA-RAG/` contains all the multi-agent RAG architectures with their respective setups. Some architectures like **RTA** only have a small config bit at the top of the script where you can choose alpha and aggregation method.
    - `help/` contains aggregations methods
* `RAG_baseline/` contains the baseline of the project. If one would want to change between the two baseline setups ($top k=1$ and $top-k = 5$ chunks), the `TOP_K` variable in `help_functions.py` in *line 17* needs to be changed. This variable decides how many chunks the retriever returns.
* `Agents.py` contains all the different agents which is used as building blocks for the architectures. The architecture scripts can then just call the different agents
* `help_functions.py` is a shared module containing utility functions used across all pipeline scripts, including the Elasticsearch client connection, question loading from CSV, BM25 retrieval, LLM API calls with retry logic, score parsing, version tracking, and answer normalization helpers for consensus and stability checks across debate rounds

---