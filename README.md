# XAI for Tsunami Time Series

Explainability analysis of a deep learning model for tsunami classification using 
time series data from buoy sensors. This repository implements and compares two 
XAI (Explainable Artificial Intelligence) methods: **ts-MULE** (Schlegel et al. 2021)
and **CONFETTI** (Cetina et al. 2026), to explain predictions of a modified version of the
model proposed by Núñez et al. (2022).

---

## 📌 Overview

This project applies XAI techniques to interpret the predictions of a 1D-CNN model 
trained with transfer learning from a model proposed by Núñez et al (2022), 
to classify tsunami events from multivariate time series recorded by a network 
of 6 buoys. The analysis identifies which temporal regions and sensors are most relevant 
for the model's decisions.

---

## 🌊 Methods

### ts-MULE
Local perturbation-based explainability method adapted for time series. Generates 
relevance scores for each timestep and feature by fitting a local linear model 
(Lasso regression) over perturbed samples (Schlegel et al. 2021).

### CONFETTI
Counterfactual explanation method for time series. Generates counterfactual instances 
— minimal modifications to the input that would change the model's prediction — 
using a genetic algorithm (Cetina et al. 2026).

---

## 🗂️ Repository Structure

```
├── xai_utils/                  # Main XAI utility module
│   ├── __init__.py
│   ├── tsmule_analysis.py      # ts-MULE analysis function
│   ├── tsmule_plots.py         # ts-MULE plotting functions
│   ├── tsmule_convergence.py   # ts-MULE convergence analysis
│   ├── confetti_analysis.py    # CONFETTI analysis function
│   ├── confetti_plots.py       # CONFETTI plotting functions
│   ├── confetti_convergence.py # CONFETTI convergence analysis
│   └── comparison_plots.py     # Comparative plotting functions
│
├── 0X_name.ipynb               # Introductory files, data and model setup
├── 1X_name.ipynb               # CONFETTI implementation (Cetina et al. 2026)
├── 2X_name.ipynb               # ts-MULE implementation (Schlegel et al. 2021)
├── 3X_name.ipynb               # Joint analysis using both methods
│
├── DATA/                       # Dataset (see Data section for download)
├── MODELS/                     # Trained models (see Installation for download)
├── requirements.txt            # Project dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/username/xai-tsunami-ts.git
cd xai-tsunami-ts
```

2. Create and activate a conda environment:
```bash
conda create -n xai_tsunami python=3.12
conda activate xai_tsunami
```

3. Download the required packages (modified versions with bug fixes):

| Package | Download | Description |
|:---|:---:|:---|
| ts-MULE | [Download](https://drive.google.com/drive/folders/1kvca5dyYCNOYtVMljw_yGAYRRfMGBfze?usp=drive_link) | Modified version with bug fixes |
| CONFETTI | [Download](https://drive.google.com/drive/folders/1PipK4KWZqMZ6Ua5apq4kpOO7e0c_0M9j?usp=drive_link) | Original package |

   Once downloaded, follow the installation instructions for each package:

**ts-MULE** — No installation required. Download the folder and place it 
   in `METHODS/ts-mule/` at the root of the repository. Then add it to the 
   Python path at the beginning of each notebook:
```python
   import sys
   sys.path.insert(0, 'METHODS/ts-mule')
```

   **CONFETTI** — Install from its local directory:
```bash
   cd path/to/confetti
   pip install confetti-ts
```

4. Install remaining dependencies:
```bash
pip install -r requirements.txt
```
---

## 🚀 Usage

```python
from xai_utils import analyze_with_tsmule, analyze_with_confetti
from xai_utils import plot_relevance_map, plot_best_counterfactual
from xai_utils import plot_comparison_summary

# ts-MULE analysis
resultados_tsmule = analyze_with_tsmule(
    model=model,
    x=instance,
    n_runs=300,
    feature_names=['Buoy 1', 'Buoy 2', 'Buoy 3', 'Buoy 4', 'Buoy 5', 'Buoy 6']
)

# CONFETTI analysis
resultados_cf = analyze_with_confetti(
    model_path_wrapped='MODELS/model_wrapped.keras',
    model_path_original='MODELS/model.keras',
    instance=instance,
    X_train=X_train,
    population_size=100,
    maximum_number_of_generations=200
)

# Comparative visualization
plot_comparison_summary(resultados_tsmule, resultados_cf)
```

---

## 📊 Suggested configuration

| Method | Optimal Parameters | Computation Time |
|:---:|:---:|:---:|
| ts-MULE | n_runs=300, n_samples=100 | ~9.8 min |
| CONFETTI | population_size=100, max_generations=200 | between 9.0 and 45 min* |

*Computation time is strongly dependent on the complexity of the instance being analysed

---

## 🐛 Bug Fixes in ts-MULE

The following bugs were identified and corrected from the original ts-MULE repository. 
The modified version available for download in this repository already incorporates 
all these fixes.

| # | File | Function | Bug | Fix |
|:---:|:---|:---|:---|:---|
| 1 | `tsmule/xai/lime.py` | `LimeBase.explain()` | `segmentation_method='slope-max'` — incorrect method name | Changed to `'slopes-sorted'` |
| 2 | `tsmule/xai/lime.py` | `LimeBase._explain()` | `z_hat` shape `(n,1)` instead of `(n,)` caused null Lasso coefficients | Added `.flatten()` to `z_hat` |
| 3 | `tsmule/xai/lime.py` | `LimeTS.__init__()` | `Lasso(alpha=0.01)` too aggressive — produced almost all zero coefficients | Reduced to `alpha=0.0001` |
| 4 | `tsmule/xai/evaluation.py` | `PerturbationBase.mask_percentile()` | Strict `>` operator excluded values exactly equal to percentile, producing all-ones masks | Changed to `>=` with `method='lower'` |
| 5 | `tsmule/xai/evaluation.py` | `PerturbationBase._randomize()` | `n_ons` could become negative when `n_offs * (1 + delta) > n_steps` | Added `np.clip(n_offs, 0, n_steps)` |

---
## 📁 Data

The dataset used in this project consists of synthetic tsunami time series 
generated from numerical simulations, recorded by a network of 6 virtual buoys.

| Dataset | Download | Description |
|:---|:---:|:---|
| Data set | [Download](https://drive.google.com/drive/folders/1D_eD67j7sEbZy7gyZfP9cTYWzXqrEuoh?usp=drive_link) | X_train, y_train, X_val, y_val, X_test, y_test |


Once downloaded, place the files in a `DATA/` folder at the root of the repository:
```
├── DATA/
│   ├── X_train_new.pickle
│   ├── y_train_new.pickle
│   ├── X_val_new.pickle
│   ├── y_val_new.pickle
│   ├── X_test_new.pickle
│   └── y_test_new.pickle
```

## 📋 Requirements

- Python 3.12
- TensorFlow / Keras 2.21
- ts-MULE
- CONFETTI
- NumPy, Pandas, Matplotlib, Scikit-learn, SciPy

See `requirements.txt` for full dependency list.

---

## 👤 Authors

- **Matías Sifón, MSc** — PhD Student
- **Tomás Mercado** — MSc Student
- **Ignacio Muñoz** — B.Sc. Student
- **Raquel Pezoa, PhD** — Supervisor

**Institution:** Universidad Técnica Federico Santa María  
**Course:** INF473 - Introducción a XAI

---

## 📄 License

This project is for academic purposes only.

## 📚 References

- Cetina, A. G. P., Benguessoum, K., Lourenço, R., & Kubler, S. (2026). Counterfactual 
Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series 
Classification. *Proceedings of the AAAI Conference on Artificial Intelligence*, 
17393–17400. 
[https://arxiv.org/abs/2511.13237](https://arxiv.org/abs/2511.13237)

- Núñez, J., Catalán, P. A., Valle, C., Zamora, N., & Valderrama, A. (2022). 
Discriminating the occurrence of inundation in tsunami early warning with 
one-dimensional convolutional neural networks. *Scientific Reports*, 12(1). 
[https://doi.org/10.1038/s41598-022-13788-9](https://doi.org/10.1038/s41598-022-13788-9)

- Schlegel, U., Lam, D. V., Keim, D. A., & Seebacher, D. (2021). TS-MULE: Local 
Interpretable Model-Agnostic Explanations for Time Series Forecast Models. *Joint 
European Conference on Machine Learning and Knowledge Discovery in Databases*, 5–14. 
[https://arxiv.org/abs/2109.08438](https://arxiv.org/abs/2109.08438)


