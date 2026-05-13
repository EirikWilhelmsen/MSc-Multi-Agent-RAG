Here are all the results for the thesis
---
Generation accuracy across 500 HoH benchmark questions, classified by LLM-as-judge. Each predicted answer is categorised as Correct (matches the updated gold answer), Outdated (matches the outdated gold answer), or Wrong (matches neither). Accuracy is the proportion of Correct predictions.

| Method | Acc. | *Correct* | *Outdated* | *Wrong* |
|---|---|---|---|---|
| Baseline RAG (k=1) | 42.20% | 211 | 143 | 146 |
| Baseline RAG (k=5) | 50.80% | 254 | 127 | 119 |
| RTA ($\alpha$=0.3) | **56.40%** | 282 | 94 | 124 |
| RTA ($\alpha$=0.5) | 48.40% | 242 | 141 | 117 |
| RTA ($\alpha$=0.7) | 54.00% | 270 | 105 | 125 |
| RCA (Majority vote) | 44.80% | 224 | 153 | 123 |
| RCA (Confidence Based) | 53.20% | 266 | 101 | 133 |
| RCA (Random) | 43.60% | 218 | 104 | **178** |
| RCA_T (Majority vote) | 56.20% | 281 | 103 | 116 |
| RCA_T (Confidence based) | **56.40%** | 282 | 91 | 127 |
| RCA_T (Random) | 43.00% | 215 | **114** | **171** |
| RCO (t=1) | 55.20% | 276 | 91 | 133 |
| RCO (t=0.7) | **56.40%** | 282 | 92 | 126 |
| RCDS | 54.80% | 274 | 103 | 123 |
| RCDS_D | 53.00% | 265 | 103 | 132 |
| RCDS_DS | 52.20% | 261 | 103 | 136 |

---

Generation accuracy across 500 HoH benchmark questions, classified by exact string match against the updated and outdated gold answers. Each predicted answer is categorised as Correct, Outdated, or Wrong. The last column counts questions that could not be classified by exact match. Accuracy is the proportion of Correct predictions among the matchable questions.

| Method | Acc. | *Correct* | *Outdated* | *Wrong* | *questions not exact matchable* |
|---|---|---|---|---|---|
| Baseline RAG (k=1) | 25.91% | 107 | 215 | 91 | 87 |
| Baseline RAG (k=5) | 33.41% | 138 | 177 | 98 | 87 |
| RTA ($\alpha$=0.3) | 42.62% | 176 | 127 | 110 | 87 |
| RTA ($\alpha$=0.5) | 41.11% | 171 | 150 | 95 | 84 |
| RTA ($\alpha$=0.7) | 39.86% | 167 | 164 | 88 | 81 |
| RCA (Majority vote) | 31.16% | 129 | 199 | 86 | 86 |
| RCA (Confidence based) | 32.13% | 133 | 194 | 87 | 86 |
| RCA (Random) | 27.36% | 116 | 127 | **181** | 76 |
| RCA_T (Majority vote) | 40.98% | 168 | 139 | 103 | 90 |
| RCA_T (Confidence based) | **44.39%** | 182 | 129 | 99 | 90 |
| RCA_T (Random) | 25.18% | 107 | 121 | **197** | 75 |
| RCO (t=1) | 42.44% | 174 | 120 | 116 | 90 |
| RCO (t=0.7) | 43.55% | 179 | 117 | 115 | 89 |
| RCDS | 36.47% | 151 | 159 | 104 | 86 |
| RCDS_D | 36.43% | 149 | 153 | 107 | 91 |
| RCDS_DS | 39.23% | 162 | 135 | 116 | 87 |

---

Exposure and selection recall (%) for Updated and Outdated gold-article snapshots across 500 questions. Exposure recall measures the proportion of questions where the snapshot reaches the architecture's working set; selection recall measures the proportion where it is chosen as the source of the predicted answer. Baseline (k=5) has a very high percentage of selected updated because unlike all the other architectures, baseline k=5's response is based on all 5 allocated chunks.

| Method | Exposure *Updated* | Exposure *Outdated* | Selection *Updated* | Selection *Outdated* |
|---|---|---|---|---|
| Baseline RAG (k=1) | -- | -- | 28.4 | 50.8 |
| Baseline RAG (k=5) | -- | -- | 87.6 | 87.8 |
| RTA ($\alpha$=0.3) | -- | -- | 45.0 | 30.0 |
| RTA ($\alpha$=0.5) | -- | -- | 43.8 | 34.2 |
| RTA ($\alpha$=0.7) | -- | -- | 42.2 | 37.8 |
| RCA (Confidence based) | 82.2 | 83.4 | 36.4 | 45.0 |
| RCA (Majority vote) | 82.2 | 83.4 | 33.8 | 46.0 |
| RCA (Random) | 82.2 | 83.4 | 32.8 | 31.8 |
| RCA_T (Confidence based) | 82.2 | 83.4 | 47.4 | 30.6 |
| RCA_T (Majority vote) | 82.2 | 83.4 | 43.4 | 35.0 |
| RCA_T (Random) | 82.2 | 83.4 | 30.2 | 32.0 |
| RCO (t=1) | 57.8 | 59.4 | 47.4 | 32.0 |
| RCO (t=0.7) | 57.4 | 59.6 | 48.8 | 30.6 |
| RCDS | 82.2 | 83.4 | 40.5 | 37.6 |
| RCDS_D | 82.2 | 83.4 | 41.2 | 36.4 |
| RCDS_DS | 82.2 | 83.4 | 41.4 | 33.6 |