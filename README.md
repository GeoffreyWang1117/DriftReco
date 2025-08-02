# 📘 DriftRec: Uncertainty-Aware and Explainable Recommender System Framework

---

## 🧭 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Explainability Modules](#explainability-modules)
- [Experimental Results](#experimental-results)
- [Algorithm Comparison & Insights](#algorithm-comparison--insights)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## 🧩 Overview

**DriftRec** is a modular framework for analyzing uncertainty and explainability in recommender systems. It supports plug-and-play models like DeepFM and AutoInt, combined with SHAP and Fisher Information analysis modules to better understand model robustness under drift conditions.

---

## 🎯 Motivation

Modern recommender systems often suffer from:
- Performance degradation under feature or behavior drift
- Lack of interpretability in prediction outcomes
- Inadequate handling of cold-start and sparse features

**DriftRec** aims to bridge this gap by providing:
- Modular implementations of baseline CTR models
- Uncertainty and robustness analysis tools
- Visual insight into model decision-making processes

---

## ✨ Key Features

- ✅ End-to-end pipeline for recommendation + explainability
- ✅ Integrated SHAP value analysis for global and local interpretation
- ✅ Fisher Information Matrix for feature-level robustness estimation
- ✅ Support for Amazon Reviews 2023 (All_Beauty) dataset
- ✅ Easily extensible to additional models (DIN, xDeepFM, LLMs, etc.)

---

## 🧱 Project Structure

```bash
DriftRec/
├── data/           # Data loading and preprocessing
├── models/         # DeepFM, AutoInt model definitions
├── training/       # Training scripts
├── analysis/       # SHAP + Fisher analysis
├── notebooks/      # Jupyter notebooks for exploration
├── results/        # Output images, plots
└── README.md
```

---

## 🚀 Quick Start

```bash
# Step 1: Clone the repo
git clone https://github.com/yourname/DriftRec.git
cd DriftRec

# Step 2: Create Conda environment
conda env create -f environment.yaml
conda activate driftrec

# Step 3: Preprocess data
python data/preprocess.py

# Step 4: Train model
python training/step3_train_deepfm_ms.py

# Step 5: Run analysis
python analysis/step4_shap_analysis.py
python analysis/step4_fisher_analysis.py
```

---

## 🧠 Supported Models

| Model     | Feature Interaction       | Interpretability | Use Case            |
|-----------|---------------------------|------------------|---------------------|
| DeepFM    | FM + MLP                  | Medium           | Sparse CTR datasets |
| AutoInt   | Self-attention-based      | High             | Cold-start & drift  |
| (planned) xDeepFM | CIN + DNN         | Medium           | High-order modeling |

---

## 🔍 Explainability Modules

### SHAP Analysis
- Measures per-feature contribution to model prediction
- Visualizes both local (single sample) and global importance

### Fisher Information Matrix
- Estimates feature sensitivity and model robustness
- Useful for detecting potential overfitting and drift-prone features

*(Visualizations will be shown here, e.g. SHAP summary plots and Fisher bar charts)*

---

## 📊 Experimental Results

| Model     | AUC   | LogLoss | Notes                  |
|-----------|-------|---------|------------------------|
| DeepFM    | 0.742 | 0.511   | Baseline               |
| AutoInt   | 0.755 | 0.498   | Strong under cold-start|

*(Insert SHAP and Fisher visualization images under `/results` folder)*

---

## 📘 Algorithm Comparison & Insights

```markdown
| Model     | Interaction Method | Interpretability | Training Cost | Notes                      |
|-----------|--------------------|------------------|----------------|----------------------------|
| DeepFM    | FM + DNN           | Medium           | Medium         | Good for sparse features   |
| AutoInt   | Attention layers   | High             | High           | Better at structure learning |
| xDeepFM   | CIN + DNN          | Medium           | High           | Planned extension          |
```

---

## 🛠️ Future Work

- [ ] Add DIN and xDeepFM as baselines
- [ ] Integrate FAISS-based retrieval for hybrid ranking
- [ ] Add prompt-based explanation using open LLMs
- [ ] Compare robustness under temporal drift
- [ ] Extend to MIND or Criteo datasets

---

## 📚 Citation

```bibtex
@misc{driftrec2025,
  author = {Your Name},
  title = {DriftRec: Uncertainty-Aware and Explainable Recommender Framework},
  year = {2025},
  note = {GitHub project},
  url = {https://github.com/yourname/DriftRec}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
