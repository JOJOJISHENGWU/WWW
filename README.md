<div align="center">

# 🚦 SCOUT
**Spectral-Community Orchestrated Unified Tri-Modal Network for Traffic Prediction**


[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## 💡 Overview

Urban traffic dynamics inherently manifest as a complex multimedia system, where **numerical rhythms**, **visual morphology**, and **linguistic semantic events** are deeply intertwined. However, existing methods rigidly treat traffic forecasting as a uni-modal graph property, leading to severe performance drops during sudden congestion or event-driven anomalies.

**SCOUT** is an innovative framework that breaks the single-modality limitation. We elevate traffic prediction into a **Tri-modal Representation Learning** paradigm by explicitly decoupling stationary rhythmic estimation from bounded morphological and semantic dynamic corrections. 

SCOUT achieves new State-of-the-Art (SOTA) results across 6 real-world benchmark datasets (including a remarkable **28.09% MAE reduction** on PeMS-BAY).

<p align="center">
  <img src="https://github.com/JOJOJISHENGWU/SCOUT/blob/main/SCOUT.png" alt="SCOUT Architecture" width="85%">
  <br>
  <em>Figure 1: The Overall Architecture of SCOUT.</em>
</p>

## 🚀 Key Modules & Design Philosophy

SCOUT explicitly rejects the "blindly stacking Transformers/Mambas" approach. Instead, it follows a strict physical topology -> morphology -> semantic logic workflow. Here is a concise introduction to each core module:

*   **1. FCR-ST (Frequency-Community Routed Spatio-Temporal Backbone)** 🎶
    *   *Introduction:* The fundamental numerical rhythm extractor. It resolves spectral-topological misalignment.
    *   *Mechanism:* Uses RFFT and Graph Laplacian K-Means to extract functional communities. Employs a **dual-kernel routing** mechanism to adaptively assign specific spatial transmission paths for high/low-frequency patterns depending on community traits.
*   **2. PCS-ViM (Propagation-Compiled Scanning VisionMamba)** 👁️
    *   *Introduction:* The geometric flow morphology corrector. It tackles the fatal "raster-scan mismatch" in gridless sensor graphs.
    *   *Mechanism:* Reorders nodes into a **diffusion-consistent sequence** based on travel-time priors, allowing a dual-axis **VisionMamba (SSM)** to capture directional congestion shockwaves as a bounded residual correction.
*   **3. TSC (Traffic State Controller)** 🧠
    *   *Introduction:* The evidence-driven event-state controller. It compiles short-term traffic dynamics into a compact latent control state for bounded spatio-temporal calibration.
    *   *Mechanism:* Extracts a 5-dimensional evidence vector from recent traffic dynamics and induces a latent control state via a lightweight **MLP encoder** with prototype anchoring, which boundedly modulates community routing and residual correction through warmup-gated signals. An optional frozen-LLM branch (TESC-EoS) can provide additive semantic enhancement but is disabled in the main setting.

## ⚙️ Algorithm & Execution Flow

The systematic integration of these three modules is formalized in the SCOUT Training and Inference loop below, demonstrating how baseline forecasting and cross-modal residual corrections are compiled:

```text
Algorithm 1: Training and Inference Procedure of SCOUT
---------------------------------------------------------------------------
Input: Tensor X, Graph A, community map C, target Y, epochs E, warmup schedules.
Output: Prediction Y_hat.

Preprocessing (Offline):
  1. Compute Laplacian L, spectral assignments C, and node priorities q.
  2. Initialize parameters Θ; freeze LLM backbone.

Training Loop:
  For e = 1 to E do:
    For each mini-batch (X, Y) do:
      
      [Stage 1: FCR-ST] (Spectral-Topological Backbone)
        - Apply RFFT & aggregate community spectrum M_comm.
        - Extract periodic features Z via Top-k filtering and Conv2D.
        - Modulate features into H_mod via community affine recalibration.
        - Compute routing weights ρ_comm = σ(MLP(M_comm)).
        - Spatial Prop: H_spatial = ρ_comm ⊙ K_local + (1-ρ_comm) ⊙ K_global.
      
      [Stage 2: PCS-ViM] (Morphological Correction)
        - Generate pseudo-image H; reorder node axis Π = Argsort(q).
        - Extract morphology h_scan ← Dual-SSM(H^Π).
        - Compute image residual ΔY_img via prototype-guided injection.
      
      [Stage 3: TSC] (State-Guided Modulation)
        - Construct evidence vector v_t from short-window statistics.
        - Induce control state s_ctrl via MLP encoder and prototype anchoring.
        - Dynamically modulate routing weights to ρ_tilde via warmup-gated signals.
        - Compute correction residual ΔY_corr via bounded state modulation.
      
      [Optimization]
        - Aggregate baseline: Y_FCR = Linear(H_spatial).
        - Final Fusion: Y_hat = Y_FCR + λ_img(t)*ΔY_img + λ_sem(t)*ΔY_corr.
        - Update Θ by minimizing composite Loss: L_pred + λ_spec*L_spec + L_reg.

Inference: 
  - Fix λ=1, disable gradients, use O(1) cached LLM representations.
---------------------------------------------------------------------------
```

## 📊 Main Results & Datasets

SCOUT is evaluated on 6 real-world datasets in the full pipeline. In this anonymous review package, we release a **PEMS08 verification artifact** (`config + checkpoint + full train log + traceability table`) to balance reproducibility and code confidentiality.

| Datasets  | Nodes | Edges | Time Steps | Time Ranges           |
|-----------|-------|-------|------------|-----------------------|
| PeMS03    | 358   | 547   | 26208      | 9/1/2018-11/30/2018   |
| PeMS04    | 307   | 340   | 16992      | 1/1/2018-2/28/2018    |
| PeMS07    | 883   | 866   | 28224      | 5/1/2017-8/31/2017    |
| PeMS08    | 170   | 295   | 17856      | 7/1/2016-8/31/2016    |
| METR-LA   | 207   | 1515  | 34272      | 03/2012-06/2012       |
| PeMS-BAY  | 325   | 2369  | 52116      | 01/2017-05/2017       |

## 🔬 Experimental Settings

To ensure reproducibility, SCOUT is evaluated under strict experimental configurations:

- **Hardware Environment:** NVIDIA GeForce RTX 5090 (32 GB) GPU, Intel Xeon Platinum 8470Q CPU, and 90 GB RAM.
- **Dataset Split:** Chronological split of 7:2:1 for training, validation, and testing.
- **Hyperparameters:** 
  - Input sequence length $T = 12$ (60 minutes historical window).
  - Hidden dimension $d_{\text{model}} = 128$.
  - Decoder layers $L = 3$.
  - Optimizer: Adam with learning rate $1 \times 10^{-4}$ and batch size $32$.
- **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE).

## 📉 Ablation & Anomaly Stability

Why a tri-modal architecture? Our experiments explicitly demonstrate the fatal flaws of unimodal approaches during high-horizon prediction and sudden distribution shifts:

- **Without Rhythmic Routing (w/o FCR-ST):** Severe degradation in long-horizon prediction (120 mins) due to the inability to isolate local anomalies from global traffic inertia.
- **Without Visual Scanning (w/o PCS-ViM):** Significant inflation in RMSE. Standard raster scanning distorts physical propagation continuity, leading to morphological extrapolation failures.
- **Without State Control (w/o TSC):** Poor adaptation to abrupt volatility (shock-to-recovery phases). TSC explicitly acts as an evidence-driven state anchor, generating a **3.62% window-level MAE reduction** during sudden anomaly transitions.

## ⚡ Complexity & Efficiency

The computational complexity is optimized for real-world scaling:
1. **FCR-ST** is bottlenecked by efficient RFFT $\mathcal{O}(BNd T\log T)$ and Chebyshev filtering $\mathcal{O}(BTK|\mathcal{E}|d)$.
2. **PCS-ViM** exploits the linear scaling of State Space Models (SSMs), executing in $\mathcal{O}(BTNd)$, completely bypassing the $\mathcal{O}((TN)^2)$ quadratic cost of Vision Transformers.
3. **TSC** realizes state-guided modulation with lightweight MLP-based inference, with an optional $\mathcal{O}(1)$ overhead per sample by caching pre-computed EoS templates from a frozen LLM.
Consequently, the entire multi-modal framework scales linearly $\mathcal{O}(N \cdot T)$, demonstrating high suitability for resource-constrained edge ITS deployment.

## 🛠️ Getting Started

### 1. Prerequisites
- Linux or Windows (Tested on Ubuntu 20.04)
- Python $\ge$ 3.9
- PyTorch $\ge$ 2.0.0
- CUDA $\ge$ 11.8

```bash
# Clone the repository
git clone https://github.com/YourUsername/SCOUT.git
cd SCOUT

# Install dependencies
pip install -r requirements.txt

# Install Vision Mamba extensions (mamba-ssm, causal-conv1d)
pip install causal-conv1d>=1.2.0
```

### 2. PEMS08 artifact verification

```bash
python eval.py --config configs/pems08.yaml --ckpt checkpoints/pems08_best.pth
```

> Note: this anonymous repository is a method-level pseudocode package. Numerical evaluation is only valid for checkpoints produced by the same code snapshot. If keys mismatch, evaluation stops explicitly.

## 🧩 Pseudocode Mapping to Paper (Reviewer-Facing)

To avoid ambiguity during review, `models/*.py` are organized as **paper-aligned pseudocode modules**. The implementation intentionally mirrors Algorithm 1 and the equation-level semantics in the manuscript.

### Stage-1: `models/FCR_ST.py`
- Mirrors: RFFT -> Top-$k$ periodic extraction -> community affine recalibration -> dual-kernel routing.
- Exposed symbols: `M_comm`, `rho_comm`, `H_mod`, `H_spatial`.

### Stage-2: `models/PCS_ViM.py`
- Mirrors: pseudo-image construction -> diffusion-priority sorting `Pi=Argsort(q)` -> dual-axis scan -> prototype-guided residual correction.
- Exposed symbols: `node_priority_q`, `perm_pi`, `h_scan`, `delta_y_img`.

### Stage-3: `models/TSC.py`
- Mirrors: evidence vector construction -> MLP state induction + prototype anchoring -> warmup-gated routing modulation -> bounded residual correction.
- Exposed symbols: `v_t`, `s_ctrl`, `rho_tilde`, `delta_y_corr`.

### Unified Aggregation: `models/SCOUT.py`
- Performs residual branch fusion exactly as formulated:
  `y_hat = y_fcr + lambda_img * delta_y_img + lambda_corr * delta_y_corr`

