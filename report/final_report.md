# AI Tutor for Personalized Learning Recommendations
## Final Project Report

**Course:** Artificial Intelligence Lab
**Team:** Muhammad Ibrahim (Leader), Arsal, Ali
**Date:** June 2026

---

## 1. Problem Statement

Students often struggle with knowing what to study next after a
quiz. A fixed curriculum ignores individual performance differences.
This project builds an AI Tutor that analyzes a student's quiz score,
response time, and confidence level to generate a personalized
learning recommendation — telling the student whether to review
basics, practice more, or advance to the next topic.

**Input:**
- Quiz score (%)
- Response time (seconds)
- Confidence level (High / Medium / Low)
- Topic
- Previous score (%)

**Output:**
- Recommendation (Review Basics / Practice More / Next Topic)
- Next topic suggestion
- Practice question count
- Revision flag

---

## 2. PEAS Framework

| PEAS Element | Description |
|---|---|
| **Performance** | Recommendation accuracy, student learning mastery rate |
| **Environment** | Web-based educational platform (Streamlit app) |
| **Actuators** | Recommendation output, next topic suggestion, practice plan |
| **Sensors** | Quiz score, response time, confidence level, previous score |

**Environment Classification:**
- Partially Observable (cannot directly measure understanding)
- Stochastic (same inputs may yield different outcomes due to context)
- Sequential (past performance informs current recommendation)
- Dynamic (student knowledge changes between sessions)

---

## 3. Methodology

### 3.1 Rule-Based Engine

The Rule-Based Engine applies explicit IF-THEN logic:
### Primary Rules

- Rule 1: score < 40 → Review Basics
- Rule 2: 40 <= score < 70 → Practice More
- Rule 3: score >= 70 → Next Topic

### Modifier Rules

- Modifier A: response_time > 80s
  - +3 extra practice questions (slow response)

- Modifier B: confidence == "Low"
  - +2 extra questions
  - Revision flagged

- Modifier C: score dropped > 10% from previous score
  - Revision flagged
**Advantages:** Fully transparent, no training data required,
easy to explain in a Viva.

**Disadvantages:** Rules are hand-crafted — they cannot learn
patterns beyond what was manually defined.

### 3.2 Decision Tree Classifier

A `DecisionTreeClassifier` from scikit-learn was trained on
200 synthetic student quiz records.

**Configuration:**
- `max_depth = 4` (prevents overfitting on small dataset)
- `criterion = "gini"` (standard Gini impurity)
- `random_state = 42` (reproducible results)
- 80% train / 20% test split (stratified)

**Features used for training:**
1. `score_pct` — quiz score percentage
2. `response_time` — time taken in seconds
3. `confidence_encoded` — High=2, Medium=1, Low=0
4. `prev_score` — previous quiz score

**Advantages:** Learns patterns from data, interpretable
tree structure, naturally handles non-linear boundaries.

**Disadvantages:** Requires labeled training data, may not
generalize to unseen scenarios without retraining.

---

## 4. Dataset

A synthetic dataset of 200 records was generated using
`data/generate_dataset.py`. Labels were assigned using the
same thresholds as the rule engine, with ±5% random noise
added to create soft boundaries and prevent perfect overfitting.

| Column | Type | Range |
|---|---|---|
| score_pct | float | 5.0 – 100.0 |
| response_time | float | 10.0 – 150.0 |
| confidence | string | High / Medium / Low |
| prev_score | float | 0.0 – 100.0 |
| recommendation | string | 3 classes |

**Class distribution (approximate):**

| Label | Count |
|---|---|
| Review Basics | ~70 |
| Practice More | ~80 |
| Next Topic | ~50 |

---

## 5. AI Integration

Two AI methods are integrated and compared:

| Aspect | Rule-Based Engine | Decision Tree |
|---|---|---|
| Type | Symbolic AI | Machine Learning |
| Training | None required | Supervised learning |
| Interpretability | Very high | High (tree paths) |
| Adaptability | Fixed rules | Learns from data |
| Explainability | Rule triggered shown | Split path shown |

Both methods produce the same output structure, allowing
direct comparison in the Evaluation tab.

---

## 6. Results

Evaluation was performed on a held-out 20% test set (40 records).

| Metric | Rule-Based Engine | Decision Tree |
|---|---|---|
| Accuracy | ~0.80 | ~0.85 |
| Precision | ~0.79 | ~0.85 |
| Recall | ~0.80 | ~0.85 |
| F1 Score | ~0.79 | ~0.84 |

*Note: Exact values vary slightly each run due to dataset noise.*

**Observation:** The Decision Tree achieves slightly higher
accuracy because it learns soft boundaries from data rather
than relying on fixed hard thresholds.

---

## 7. Explainability

For every recommendation, the system shows:

1. **Primary rule or Decision Tree prediction** — what fired
2. **Modifier rules applied** — what adjustments were made
3. **Step-by-step reasoning log** — every decision point
4. **Natural-language explanation** — plain-English paragraph
5. **Key factors table** — score signal, time signal, confidence,
   trend vs threshold

This satisfies the Lab Guide requirement that the system must
help users *"see and understand how the AI reached its result"*.

---

## 8. UI Overview

The Streamlit app is organized into four tabs:

| Tab | Module Covered |
|---|---|
| Get Recommendation | Problem Setup + Core Logic |
| Charts | Visual UI (4 Plotly charts) |
| Explainability | Explainability Module |
| Evaluation | Evaluation Module |

---

## 9. Limitations

1. **Synthetic dataset** — results may differ with real
   student data
2. **Small dataset** — 200 records limits model generalization
3. **No user authentication** — single session only
4. **Fixed topics** — adding new topics requires code changes
5. **No persistence** — results are not saved between sessions

---

## 10. Future Improvements

1. Integrate a real student database (SQL/PostgreSQL)
2. Add spaced repetition scheduling (forgetting curve)
3. Use a larger, real-world educational dataset
4. Add multi-student support with login
5. Replace Decision Tree with Random Forest for better accuracy
6. Add NLP-based question generation for practice problems

---

## 11. Conclusion

This project successfully demonstrates a working AI Tutor
that combines Rule-Based and Machine Learning AI approaches
to generate personalized learning recommendations. The system
satisfies all five modules required by the AI Lab Guide,
uses only the recommended tech stack, and produces explainable,
visual, and evaluable results — meeting the full criteria
for the Final Year BS CS AI Lab Project.

---

## References

1. Russell, S., & Norvig, P. (2022). *Artificial Intelligence:
   A Modern Approach* (4th ed.). Pearson.
2. Scikit-learn documentation — DecisionTreeClassifier.
   https://scikit-learn.org
3. Streamlit documentation. https://docs.streamlit.io
4. Plotly Python documentation. https://plotly.com/python  