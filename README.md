# 🎓 AI Tutor — Personalized Learning Recommendations

> Final Year BS Computer Science — AI Lab Project
> Built on the PEAS Framework | Rule-Based Engine + Decision Tree Classifier

---

## 📌 Project Overview

The AI Tutor is a Streamlit web application that takes student quiz
performance data as input and outputs personalized learning
recommendations. It combines two AI approaches — a Rule-Based Engine
and a Decision Tree Classifier — and compares their performance
side by side.

The project satisfies all five modules required by the AI Lab Guide:

| Module | Description |
|---|---|
| A — Problem Setup | Input form with validation, clear output |
| B — Core Logic | Rule engine + Decision Tree, modular code |
| C — Visual UI | Bar, Pie, Line, Radar charts via Plotly |
| D — Explainability | Natural-language explanation + rule/tree path |
| E — Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |

---

## 👥 Team

| Name | Role |
|---|---|
| Muhammad Ibrahim | Team Leader, Core AI Logic, Architecture |
| Arsal | Streamlit UI, Testing, Bug Fixes |
| Ali | Dataset, Screenshots, Documentation |

---

## ✨ Features

- **Two AI Modes** — Switch between Rule-Based Engine and
  Decision Tree in the sidebar
- **Personalized Output** — Recommendation, next topic,
  practice question count, revision flag
- **4 Interactive Charts** — Bar, Pie, Line, Radar (Plotly)
- **Full Explainability** — Step-by-step reasoning log,
  key factors table, tree split path
- **Evaluation Dashboard** — Side-by-side metric comparison,
  confusion matrix heatmap, classification report
- **Retrain Button** — Retrain the Decision Tree on demand
  from the sidebar

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit | Web UI framework |
| scikit-learn | Decision Tree classifier |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical operations |
| Plotly | Interactive charts |
| Matplotlib | (available, used if needed) |

---

## 📁 Folder Structure