# 🎯 LLM Confidence Calibration & Overconfidence Analysis

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**A production-grade statistical framework for diagnosing and correcting overconfidence in Large Language Models**

[📊 Results](#-results) · [🏗 Architecture](#-architecture) · [🚀 Quick Start](#-quick-start) · [📖 Methodology](#-methodology)

</div>

---

## 🧠 Problem Statement

Large Language Models are systematically overconfident — they assign high probabilities to incorrect answers, making them unreliable for production deployment. This project builds a rigorous calibration pipeline that:

- **Quantifies** overconfidence through Expected Calibration Error (ECE)
- **Diagnoses** hallucination patterns at the logit level
- **Corrects** miscalibration via post-hoc temperature scaling
- **Validates** improvements through 1000-iteration bootstrap testing

> Evaluated on **Mistral-7B** and **Phi-2** across **500 BoolQ samples**

---

## 📊 Results

| Metric | Before Calibration | After Calibration | Improvement |
|--------|-------------------|-------------------|-------------|
| ECE (Mistral-7B) | 0.187 | 0.071 | **~62% reduction** |
| Overconfident Hallucination Rate | 18.4% | 13.6% | **4.8pp reduction** |
| Accuracy | 84.2% | 84.2% | **Zero degradation** |
| Bootstrap Stability (std) | 0.043 | 0.019 | **55% more stable** |

---

## 🏗 Architecture

### System Overview

```mermaid
graph TD
    A[BoolQ Dataset\n500 Samples] --> B[Tokenizer\nHuggingFace]
    B --> C[LLM Inference\nMistral-7B / Phi-2]
    C --> D[Logit Extraction\nToken-level probabilities]
    D --> E[Confidence Scorer\nSoftmax normalization]
    E --> F{Calibration\nDiagnostics}

    F --> G[ECE Calculator\nBin-based error]
    F --> H[Hallucination Detector\nConfidence threshold analysis]
    F --> I[Reliability Diagram\nCalibration curve]

    G --> J[Temperature Scaling\nPost-hoc correction]
    H --> J
    I --> J

    J --> K[Bootstrap Validator\n1000 iterations]
    K --> L[Calibrated Model\nProduction-ready]

    style A fill:#1a1a2e,color:#00ffc8
    style L fill:#1a1a2e,color:#00ffc8
    style J fill:#16213e,color:#fff
    style K fill:#16213e,color:#fff
```

### Calibration Pipeline

```mermaid
sequenceDiagram
    participant D as Dataset
    participant M as LLM Model
    participant E as ECE Engine
    participant T as Temperature Scaler
    participant V as Validator

    D->>M: Feed 500 BoolQ questions
    M->>E: Return logits + predictions
    E->>E: Compute confidence bins
    E->>E: Measure ECE per bin
    E->>T: Report miscalibration score
    T->>T: Optimize temperature T*
    T->>V: Apply scaled probabilities
    V->>V: Bootstrap 1000 iterations
    V->>V: Compute mean ± std ECE
    V-->>T: Validate improvement
    T-->>M: Deploy calibrated model
```

### ECE Computation Flow

```mermaid
flowchart LR
    A[Raw Logits] --> B[Softmax\nP = softmax logits]
    B --> C[Confidence Bins\n10 equal-width bins]
    C --> D[Accuracy per Bin\nacc_m = correct / total_m]
    C --> E[Confidence per Bin\nconf_m = avg confidence_m]
    D --> F[ECE Formula\nΣ bins × abs acc-conf / N]
    E --> F
    F --> G{ECE > threshold?}
    G -->|Yes| H[Apply Temperature\nScaling T*]
    G -->|No| I[Model is Calibrated ✅]
    H --> J[Re-evaluate ECE]
    J --> G

    style A fill:#0f3460,color:#fff
    style I fill:#0f3460,color:#00ffc8
```

---

## 📂 Project Structure

```
LLM-Confidence-Calibration/
│
├── 📓 notebooks/
│   └── calibration_analysis.ipynb      # Main analysis notebook
│
├── 🔬 src/
│   ├── calibration/
│   │   ├── ece_calculator.py           # ECE computation engine
│   │   ├── temperature_scaling.py      # Post-hoc temperature optimizer
│   │   └── bootstrap_validator.py      # 1000-iteration stability tester
│   │
│   ├── evaluation/
│   │   ├── hallucination_detector.py   # Overconfidence quantification
│   │   ├── reliability_diagram.py      # Calibration curve visualizer
│   │   └── model_evaluator.py          # BoolQ inference pipeline
│   │
│   └── utils/
│       ├── data_loader.py              # BoolQ dataset handler
│       └── logit_extractor.py          # Token-level probability extractor
│
├── 📊 results/
│   ├── reliability_diagrams.png        # Before/after calibration curves
│   ├── ece_comparison.png              # ECE improvement chart
│   └── bootstrap_distribution.png     # Stability analysis plot
│
├── 📋 requirements.txt
├── 🔧 config.yaml                      # Model and evaluation settings
└── 📖 README.md
```

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/debasmita30/LLM-Confidence-Calibration.git
cd LLM-Confidence-Calibration
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Calibration Pipeline
```bash
jupyter notebook notebooks/calibration_analysis.ipynb
```

---

## 📖 Methodology

### 1. Logit-Level Confidence Extraction

Raw logits are extracted before softmax normalization, giving direct access to the model's internal confidence distribution across the vocabulary:

```python
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values
```

### 2. Expected Calibration Error (ECE)

ECE measures the gap between predicted confidence and actual accuracy across probability bins:

```
ECE = Σ (|B_m| / n) × |acc(B_m) − conf(B_m)|
```

Where:
- `B_m` = samples in bin m
- `acc(B_m)` = accuracy within bin
- `conf(B_m)` = average confidence within bin

### 3. Temperature Scaling

A single scalar parameter `T` is optimized on a held-out validation set to correct the confidence distribution without retraining:

```python
calibrated_probs = softmax(logits / T*)
```

`T*` is found by minimizing Negative Log-Likelihood on the validation set.

### 4. Bootstrap Validation

1000 bootstrap iterations with replacement validate that improvements are statistically stable and not artifacts of the test sample:

```python
bootstrap_eces = []
for _ in range(1000):
    sample = resample(test_data)
    bootstrap_eces.append(compute_ece(sample))

mean_ece = np.mean(bootstrap_eces)
std_ece  = np.std(bootstrap_eces)
```

---

## 🔬 Models Evaluated

| Model | Parameters | Architecture | Base ECE | Calibrated ECE |
|-------|-----------|--------------|----------|----------------|
| Mistral-7B | 7B | Decoder-only | 0.187 | 0.071 |
| Phi-2 | 2.7B | Decoder-only | 0.164 | 0.063 |

---

## 📈 Key Findings

- **Larger models are not better calibrated** — Mistral-7B had higher ECE than Phi-2 despite more parameters
- **Temperature scaling is highly effective** — single scalar achieves 62% ECE reduction with zero accuracy cost
- **Overconfidence clusters in high-confidence bins** — 80%+ of hallucinations occur when model confidence exceeds 90%
- **Bootstrap confirms stability** — improvements hold with std < 0.02 across 1000 iterations

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Inference | HuggingFace Transformers |
| Deep Learning | PyTorch 2.0+ |
| Statistical Analysis | SciPy, NumPy |
| Dataset | BoolQ (Google Research) |
| Visualization | Matplotlib |
| Notebook | Jupyter |

---

## 🔭 Future Work

- [ ] Extend to GPT-4 and Claude via API-level confidence proxies
- [ ] Implement Platt Scaling and Isotonic Regression as alternatives
- [ ] Multi-class calibration beyond binary BoolQ
- [ ] Real-time calibration monitoring dashboard
- [ ] Integration with LLM evaluation frameworks (LM-Eval-Harness)

---

## 👩‍💻 Author

<div align="center">

**Debasmita Chatterjee**

AI Engineer · LLM Evaluation · Calibration Systems

[![GitHub](https://img.shields.io/badge/GitHub-debasmita30-181717?style=flat-square&logo=github)](https://github.com/debasmita30)


</div>

---

<div align="center">
⭐ If this helped your research, star the repo
</div>
