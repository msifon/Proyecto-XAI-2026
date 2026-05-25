# XAI for Tsunami Time Series

Explainability analysis of a deep learning model for tsunami classification using 
time series data from buoy sensors. This repository implements and compares two 
XAI (Explainable Artificial Intelligence) methods: **ts-MULE** and **CONFETTI**.

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
(Lasso regression) over perturbed samples.

### CONFETTI
Counterfactual explanation method for time series. Generates counterfactual instances 
— minimal modifications to the input that would change the model's prediction — 
using a genetic algorithm.

---

## 🗂️ Repository Structure
├── xai_utils/                  # Main XAI utility module
│   ├── init.py
│   ├── tsmule_analysis.py      # ts-MULE analysis function
│   ├── tsmule_plots.py         # ts-MULE plotting functions
│   ├── tsmule_convergence.py   # ts-MULE convergence analysis
│   ├── confetti_analysis.py    # CONFETTI analysis function
│   ├── confetti_plots.py       # CONFETTI plotting functions
│   ├── confetti_convergence.py # CONFETTI convergence analysis
│   └── comparison_plots.py     # Comparative plotting functions
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_tsmule_analysis.ipynb
│   ├── 02_confetti_analysis.ipynb
│   ├── 03_convergence_tsmule.ipynb
│   ├── 04_convergence_confetti.ipynb
│   └── 05_comparison.ipynb
│
├── requirements.txt            # Project dependencies
├── .gitignore
└── README.md


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
| ts-MULE | [Download]([https://drive.google.com/your-link-here](https://drive.google.com/drive/folders/1kvca5dyYCNOYtVMljw_yGAYRRfMGBfze?usp=drive_link)) | Modified version with bug fixes |
| CONFETTI | [Download]([https://drive.google.com/your-link-here](https://drive.google.com/drive/folders/1PipK4KWZqMZ6Ua5apq4kpOO7e0c_0M9j?usp=drive_link)) | Original package |

   Once downloaded, install each package from its local directory:
```bash
# Install ts-MULE
cd path/to/ts-mule
pip install -e .

# Install CONFETTI
cd path/to/confetti
pip install -e .
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

## 📊 Key Results

| Method | Optimal Parameters | Computation Time |
|:---:|:---:|:---:|
| ts-MULE | n_runs=300, n_samples=100 | ~9.8 min |
| CONFETTI | population_size=100, max_generations=200 | ~9.0 min |

---

## 🐛 Bug Fixes in ts-MULE

During implementation, the following bugs were identified and corrected in the 
original ts-MULE repository:

1. `segmentation_method='slope-max'` → incorrect name, corrected to `'slopes-sorted'`
2. `z_hat` without flatten in `_explain()` → shape `(n,1)` instead of `(n,)` caused null coefficients
3. `Lasso(alpha=0.01)` too aggressive → reduced to `alpha=0.0001`
4. `mask_percentile()` used strict `>` → changed to `>=` with `method='lower'`

---
## 📁 Data

The dataset used in this project consists of synthetic tsunami time series 
generated from numerical simulations, recorded by a network of 6 virtual buoys.

| Dataset | Download | Description |
|:---|:---:|:---|
| Data set | [Download]([https://drive.google.com/your-link-here](https://drive.google.com/drive/folders/1D_eD67j7sEbZy7gyZfP9cTYWzXqrEuoh?usp=drive_link)) | X_train, y_train, X_val, y_val, X_test, y_test |


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

- **Matías Sifón** — PhD Student
- **Tomás Mercado** — MSc Student
- **Ignacio Muñoz** — B.Sc. Student
- **Raaquel Pezoa** — Supervisor

**Institution:** Universidad Técnica Federico Santa María  
**Course:** INF473 - Introducción a XAI

---

## 📄 License

This project is for academic purposes only.
