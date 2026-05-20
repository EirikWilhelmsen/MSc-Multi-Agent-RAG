Here are all the results for the thesis
---
Generation accuracy across 500 HoH benchmark questions, classified by LLM-as-judge. Each predicted answer is categorised as Correct (matches the updated gold answer), Outdated (matches the outdated gold answer), or Wrong (matches neither). Accuracy is the proportion of Correct predictions.

| Method | Acc. | *Correct* | *Outdated* | *Wrong* |
|---|---|---|---|---|
| Baseline RAG (k=1) | 33.6% | 168 | 134 | 198 |
| Baseline RAG (k=3) | 42.2% | 211 | 110 | 179 |
| Baseline RAG (k=5) | 42.8% | 214 | 108 | 178 |
| RTA (a=0.3) | 50.4% | 252 | 76 | 172 |
| RTA (a=0.5) | 48.4% | 242 | 94 | 164 |
| RTA (a=0.7) | 47.4% | 237 | 105 | 158 |
| RCA (Majority vote) | 39.2% | 196 | 141 | 163 |
| RCA (Confidence based) | 39.2% | 196 | 133 | 171 |
| RCA (Random) | 35.0% | 175 | 90 | 235 |
| RCA_T (Majority vote) | 49.4% | 247 | 99 | 154 |
| RCA_T (Confidence based) | **53.0%** | 265 | 86 | 149 |
| RCA_T (Random) | 33.4% | 167 | 75 | 258 |
| RCO (t=1) | 49.6% | 248 | 86 | 166 |
| RCO (t=0.7) | 51.6% | 258 | 82 | 160 |
| RCDS | 43.8% | 219 | 111 | 170 |
| RCDS_D | 44.0% | 220 | 109 | 171 |
| RCDS_TS | 47.2% | 236 | 83 | 181 |

---

Generation accuracy across 500 HoH benchmark questions, classified by exact string match against the updated and outdated gold answers. Each predicted answer is categorised as Correct, Outdated, or Wrong. The last column counts questions that could not be classified by exact match. Accuracy is the proportion of Correct predictions among the matchable questions.

| Method | Acc. | *Correct* | *Outdated* | *Wrong* | *questions not exact matchable* |
|---|---|---|---|---|---|
| Baseline RAG (k=1) | 25.65% | 108 | 193 | 120 | 79 |
| Baseline RAG (k=3) | 33.01% | 136 | 157 | 119 | 88 |
| Baseline RAG (k=5) | 33.41% | 138 | 157 | 118 | 87 |
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
| Baseline RAG (k=1) | 28.4 | 50.8 | 27.6 | 48.2 |
| Baseline RAG (k=3) | 82.8 | 83.2 | 35.6 | 38.6 |
| Baseline RAG (k=5) | 87.6 | 87.8 | 36.2 | 38.6 |
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