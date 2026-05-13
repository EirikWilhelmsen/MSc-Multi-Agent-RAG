Contents of this folder include files and scripts that directly affect the dataset.<br>
---
Here is a description of the files inside this folder:
* `500Q/` folder contains the 500 random selected questions for testing and is structured with
    - `qid`: id of the entry
    - `pageid`: id of the *gold page*
    - `title`: title of the *gold page*
    - `question`: question
    - `current_time`: *gold page* updated version timestamp
    - `outdated_info_dates`: *gold page* outdated version timestamp
    - `answer`: answer to question
    - `outdated_answer`: outdated answer to question
* `clean_HoH_dataset/` contains a parquet file with the dataset created by Ouyang et al[$^1$].
* `data_scripts/` contains scripts used to directly on the dataset for either preprocessing purposes or information extraction
    - `count_files.py` is strictly for counting files and ensuring the dataset is correctly implemented on local device
    - `Create_500_Q.py` used to create the random selected questions for testing
    - `data_preprocessing.py` takes the raw KB containing several thousand files from different snapshot dates and process them by removing references, comments, links, etc. Eventually write to `KB_cleaned` and keeps tracks on encountered errors
    - `elasticsearch_indexing.py` indexes the cleaned KB into a single Elasticsearch index (`wikipedia_snapshots`) by chunking each article into 256-token segments with 32-token overlap and bulk-uploading them with `pageid`, `date`, and `chunk_index` as metadata
    - `extract_ids.py` reads the HoH parquet file and extracts all unique document IDs along with their associated timestamps (both updated and outdated versions), writing the result to `doc_times.json` as a mapping from `pageid` to a sorted list of dates
* `doc_times.json` displays the totalt amount of unique `page_ids` with corresponding snapshot dates
* `failed_snapshots.jsonl` is a record of all snapshots that was not able to be retrieved from wikipedia
---

[$^1$]: https://arxiv.org/abs/2503.04800
