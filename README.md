#  LLM Cost–Performance Optimization Platform

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![Monte Carlo](https://img.shields.io/badge/Monte_Carlo-Simulation-orange?style=flat-square)]()
[![Pareto](https://img.shields.io/badge/Pareto-Frontier-green?style=flat-square)]()
[![SLA](https://img.shields.io/badge/SLA-Constrained-red?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**A production-grade, multi-objective optimization system for evaluating LLM deployment strategies under cost and SLA constraints**

🌐 **[Live Demo](https://go-job-queue.vercel.app/)** &nbsp;|&nbsp; ⚙️ **[API Health](https://go-job-queue.onrender.com/health)** &nbsp;|&nbsp; 📊 **[Live Stats](https://go-job-queue.onrender.com/api/v1/stats)** &nbsp;|&nbsp; 💻 **[Source Code](https://github.com/debasmita30/go-job-queue)**

> **Note:** Hosted on Render free tier — if the demo shows "Server Offline", open the [API Health](https://go-job-queue.onrender.com/health) link first and wait 30–60 seconds for it to wake up, then refresh the demo.

</div>

---

## 🧠 Problem Statement

Deploying LLMs in production requires making simultaneous tradeoffs across five competing dimensions:

| Dimension | Challenge |
|-----------|-----------|
| **Accuracy** | Higher accuracy models cost more per request |
| **API Cost** | Budget constraints limit model selection |
| **Latency** | Low-latency requirements rule out large models |
| **SLA Compliance** | Enterprise SLAs impose hard latency ceilings |
| **Infrastructure Budget** | Monthly cost projections must stay within bounds |

This platform provides a **mathematically rigorous, simulation-driven framework** to make LLM deployment decisions that are optimal, defensible, and deployment-ready.

---

## 🌐 Live Demo

[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge&logo=render)](https://llm-cost-performance-optimization-platform.streamlit.app/)

---

## 📊 Key Results

| Optimization Method | Best Accuracy | Min Cost/req | SLA Compliance | Efficiency Score |
|--------------------|--------------|--------------|----------------|-----------------|
| Baseline (no opt.) | 91.2% | $0.0042 | 78.4% | 217 |
| Single-objective (λ=0.3) | 89.7% | $0.0021 | 84.1% | 427 |
| **Pareto-optimal** | **88.9%** | **$0.0018** | **91.3%** | **494** |
| SLA-constrained | 87.4% | $0.0019 | **100%** | 460 |

---

## 🏗 Architecture

### System Overview

```mermaid
graph TD
    A[LLM Configuration Space\nModels × Prompts × Temperature] --> B[Grid Search Engine\nExhaustive parameter sweep]
    B --> C[Simulation Engine\nToken · Cost · Latency modeling]
    C --> D{Multi-Objective\nOptimizer}

    D --> E[Pareto Frontier\nExtractor]
    D --> F[SLA Constraint\nFilter]
    D --> G[Lambda Tradeoff\nSolver]

    E --> H[Optimal Configuration\nSet]
    F --> H
    G --> H

    H --> I[Monte Carlo\nValidator\n20 runs]
    I --> J[Deployment Cost\nForecaster]
    J --> K[Executive Dashboard\nStreamlit + Plotly]

    style A fill:#1a1a2e,color:#00ffc8
    style K fill:#1a1a2e,color:#00ffc8
    style D fill:#16213e,color:#fff
    style I fill:#16213e,color:#fff
```

### Optimization Pipeline

```mermaid
sequenceDiagram
    participant C as Config Space
    participant G as Grid Search
    participant P as Pareto Solver
    participant S as SLA Filter
    participant M as Monte Carlo
    participant D as Dashboard

    C->>G: Model × Prompt × Temperature combos
    G->>G: Evaluate accuracy, cost, latency per config
    G->>P: Pass full results matrix
    P->>P: O(n log n) non-dominated sort
    P->>S: Return Pareto-optimal configs
    S->>S: Filter latency ≤ SLA threshold
    S->>S: Re-optimize under constraint
    S->>M: Pass SLA-compliant configs
    M->>M: 20-run stochastic noise simulation
    M->>M: Compute mean ± std per model
    M->>D: Validated stable configurations
    D-->>D: Render interactive analytics
```

### Dual-Objective Optimization Flow

```mermaid
flowchart LR
    A[Raw Config\nAccuracy · Cost · Latency] --> B[Normalize\nMin-Max scaling]
    B --> C[Single Objective\nObj = Acc − λ × Cost]
    B --> D[Dual Objective\nObj = Acc − λ₁Cost − λ₂Latency]
    C --> E[Lambda Sweep\nλ ∈ 0.0 → 1.0]
    D --> F[Weight Grid\nλ₁ × λ₂ combinations]
    E --> G[Optimal λ*]
    F --> G
    G --> H[Best Configuration\nper objective weight]
    H --> I{SLA Check\nLatency ≤ threshold?}
    I -->|Pass| J[Deploy ✅]
    I -->|Fail| K[Re-optimize\nunder constraint]
    K --> I

    style A fill:#0f3460,color:#fff
    style J fill:#0f3460,color:#00ffc8
```

---

## 📂 Project Structure

```
LLM-Cost-Performance-Optimization-Platform/
│
├── ⚙️ config.py                        # Global settings — SLA thresholds, λ weights, budget
├── 🚀 main.py                          # Entry point — runs full optimization pipeline
│
├── 🔬 optimizer/
│   ├── grid_search.py                  # Exhaustive config space evaluation
│   └── pareto.py                       # O(n log n) Pareto frontier extraction
│
├── 📊 visualization/
│   ├── plots.py                        # Plotly chart generators
│   └── dashboard.py                    # Streamlit interactive dashboard
│
├── 📁 results/
│   └── simulation_results.csv          # Full optimization output cache
│
├── 📋 requirements.txt
└── 📖 README.md
```

---

## Screenshots
<img width="1918" height="847" alt="Image" src="https://github.com/user-attachments/assets/25671a7f-31fa-4c1b-8b9f-fc55430f07e6" />
Live dashboard connected to Snowflake.Shows 108 total configurations tested across small, medium, and large models. The system instantly identifies the optimal config — large model at prompt_length=100 with 0.99 accuracy at just $0.00176 per request. All 3 services (Backend, Snowflake, Dashboard) are online and pulling real data.
<br><br>

<img width="1918" height="872" alt="Image" src="https://github.com/user-attachments/assets/8b4ae47d-161d-4403-a410-31c8e47efbba" />

Red diamonds = the best possible configurations. Every other dot is "dominated" — meaning a red diamond exists that is both cheaper AND more accurate. Use this to instantly find which  model setup gives the most accuracy per dollar.
<br><br>

<img width="1913" height="857" alt="Image" src="https://github.com/user-attachments/assets/78a69612-cfc9-4d6a-b306-1d1bd2314880" />

As lambda increases, the system penalizes cost more heavily and switches to 
cheaper (slightly less accurate) models. The downward slope shows the real 
accuracy-cost tradeoff — not a flat line, which means the optimization is working.
<br><br>

<img width="1510" height="510" alt="Image" src="https://github.com/user-attachments/assets/bc24e56d-834f-4a77-a254-9cdae168fc2e" />
Compares small, medium, and large models across 3 dimensions at once. 
Large model covers more area = better overall. Small/medium cluster together 
showing similar cost-latency profiles.
<br><br>


<img width="1575" height="525" alt="Image" src="https://github.com/user-attachments/assets/53f735a9-5e2b-4526-b32c-21b9ff25a7d8" />
Each cell = cost per correct answer for a given prompt length and few-shot count. 
Shorter prompts with fewer examples are cheapest (top-left). 
Use this to cut costs without hurting accuracy.
<br><br>

<img width="1532" height="540" alt="Image" src="https://github.com/user-attachments/assets/db1f79bc-6a44-4d46-a57e-ba5c477a8e27" />
Only 16.7% of configurations exceed the latency limit at this SLA threshold — meaning 83.3% of setups are production-safe. Adjust the SLA slider to see how violation rate changes in real time.
<br><br>

<img width="1553" height="712" alt="Image" src="https://github.com/user-attachments/assets/2d826a53-fba9-4156-bc80-3e9af33c9b13" />
Projected monthly cost: $528 for 10,000 daily queries using the large model. Monte Carlo stability test confirms large model is most reliable (lowest std=0.022). Dual-objective optimization balances both cost and latency penalties simultaneously to find the best production config.
<br><br>


<img width="1547" height="652" alt="Image" src="https://github.com/user-attachments/assets/57c2c3c6-e3e1-44ff-9b85-2230fcc206b8" />

Best SLA-compliant config: large model, 99% accuracy at $0.00176/request.The 3D scatter lets you rotate and explore all 108 configurations across cost, latency, and accuracy simultaneously
<br><br>


<img width="1533" height="662" alt="Image" src="https://github.com/user-attachments/assets/cf39e1e2-f734-43b8-8c80-0a0ac2a65309" />

Executive Summary auto-generates the final recommendation — optimal model, efficiency score, and monthly cost projection, all pulled live from Snowflake. Risk category dropped to **Low Risk** (16.67% violation rate).
<br><br>



## ⚙️ Core Features

### 1️⃣ Multi-Objective Optimization

Two optimization formulations supported:

**Single-objective:**
```
Objective = Accuracy − λ × Cost
```

**Dual-objective:**
```
Dual Objective = Accuracy_norm − λ₁ × Cost_norm − λ₂ × Latency_norm
```

Supports adjustable λ weights, budget filtering, and hard SLA constraints.

---

### 2️⃣ Pareto Frontier Extraction

Efficient O(n log n) non-dominated sort identifies configurations where no other config is strictly better on all objectives simultaneously:

```python
def pareto_frontier(results):
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    pareto = []
    min_cost = float('inf')
    for config in sorted_results:
        if config['cost'] < min_cost:
            pareto.append(config)
            min_cost = config['cost']
    return pareto
```

---

### 3️⃣ SLA-Constrained Optimization

```python
sla_compliant = results[results['latency_ms'] <= SLA_THRESHOLD]
optimal = sla_compliant.loc[sla_compliant['objective_score'].idxmax()]
```

Outputs best SLA-compliant configuration, SLA violation rate, and risk categorization.

---

### 4️⃣ Monte Carlo Robustness Testing

```python
bootstrap_scores = []
for _ in range(20):
    noisy_accuracy = accuracy + np.random.normal(0, noise_std)
    bootstrap_scores.append(compute_objective(noisy_accuracy, cost, latency))

mean_score = np.mean(bootstrap_scores)
std_score  = np.std(bootstrap_scores)
```

---

### 5️⃣ Deployment Cost Forecasting

```
Monthly Cost = Cost_per_request × Daily_Queries × 30
```

Real-time projection with executive budgeting breakdown.

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/debasmita30/LLM-Cost-Performance-Optimization-Platform.git
cd LLM-Cost-Performance-Optimization-Platform
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Full Pipeline
```bash
python main.py
```

### 5. Launch Dashboard
```bash
streamlit run visualization/dashboard.py
```

Open in browser: `http://localhost:8501`

---

## 📈 Dashboard Modules

| Module | Description |
|--------|-------------|
| **Pareto Scatter Plot** | Interactive cost vs accuracy frontier |
| **Lambda Tradeoff Curve** | Objective score across λ sweep |
| **Model Radar Chart** | Multi-dimensional model comparison |
| **Cost-per-Correct Heatmap** | Efficiency across config combinations |
| **SLA Risk Pie Chart** | Compliance distribution across configs |
| **3D Visualization** | Cost–Latency–Accuracy surface plot |
| **Dual-Objective Output** | Optimal config under joint constraints |
| **Executive Summary Panel** | Deployment recommendation report |

---

## 📌 Key Insights

- **Larger models maximize accuracy but increase latency risk** — not always Pareto-optimal
- **Few-shot prompting has diminishing returns** — 3-shot ≈ 5-shot accuracy at lower cost
- **Temperature negatively impacts reliability** — optimal T* typically between 0.1–0.4
- **Higher λ shifts selection toward cost-efficient models** — tradeoff is non-linear
- **SLA constraints significantly alter optimal configuration** — compliance cuts candidate pool by ~40%

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Dashboard | Streamlit + Plotly |
| Optimization | NumPy, SciPy |
| Data Processing | Pandas |
| Simulation | Custom Monte Carlo engine |
| Visualization | Plotly (3D, heatmap, radar) |

---

## 🔭 Future Work

- [ ] Real API integration (OpenAI, Anthropic, Together AI)
- [ ] Bayesian optimization for hyperparameter search
- [ ] AutoML-based model selection
- [ ] Reinforcement-based prompt optimization
- [ ] GPU cost benchmarking
- [ ] Distributed deployment modeling

---

## 👩‍💻 Author

<div align="center">

**Debasmita Chatterjee**

AI Engineer · LLM Systems · Prompt Optimization

[![GitHub](https://img.shields.io/badge/GitHub-debasmita30-181717?style=flat-square&logo=github)](https://github.com/debasmita30)


</div>

---

<div align="center">
⭐ If this helped your work, star the repo
</div>
