# The Shapley Value Contribution to Explainable AI: Experiments

Code and data for the experiments in

> Chi Zhao, Jing Liu, Elena Parilina. **The Shapley value contribution to Explainable Artificial Intelligence: a comprehensive survey.**

The paper compares non-game-theoretic XAI methods (Feature Importance, Permutation Feature Importance, LIME) with Shapley value-based methods (SHAP, KernelSHAP, SamplingSHAP, FastSHAP, ShapG) on five datasets. It reports their feature rankings, running time, and explanation quality, measured by feature dropping and the weighted decreasing slope $S$.

## Repository structure

Each dataset has its own folder that holds its data, so every notebook runs from inside its folder.

| Folder | Dataset | Task | Models |
|---|---|---|---|
| `01_life_expectancy/` | Life Expectancy (WHO), 2938 × 21 | Regression ($R^2$) | LightGBM, FNN |
| `02_diabetes/` | Diabetes, 4303 × 17 | Classification (accuracy) | LightGBM, FNN |
| `03_bank_churn/` | Bank Churn, 165034 × 10 | Classification (accuracy) | LightGBM, FNN |
| `04_crime/` | Communities and Crime, 2215 × 121 | Regression ($R^2$) | Linear Regression |
| `05_mic/` | Myocardial Infarction Complications, 1700 × 111 | Classification (accuracy) | Logistic Regression |
| `fastshap/` | FastSHAP for Diabetes and Bank Churn | Classification | LightGBM, FNN |

Notebooks inside each dataset folder:

| Notebook | Content |
|---|---|
| `00_data_processing.ipynb` | Preprocessing of the raw data (where applicable) |
| `01_*_XAI.ipynb` | Model training + SHAP, KernelSHAP, SamplingSHAP, LIME, Feature Importance, Permutation FI |
| `02_*_ShapG.ipynb` | ShapG feature importance |
| `03/04_FNN_*.ipynb` | The same for the FNN model (datasets 1–3) |
| `05/06_*_compare.ipynb`, `03_LR_compare.ipynb` | Feature-dropping comparison and weighted slope $S$ |

The comparison notebooks contain the feature rankings and running times obtained by the XAI notebooks, so the comparison figures can be reproduced without re-running the expensive explainers.

## Mapping to the paper

| Paper item | Source |
|---|---|
| Fig. 1 (Life Expectancy, LightGBM feature importance) | `01_life_expectancy/01_LGB_XAI.ipynb`, `02_LGB_ShapG.ipynb` |
| Tables 1–3 (feature rankings) | `0{1,2,3}_*/01–04_*.ipynb`, `fastshap/` |
| Tables 4–6 (running time) | `0{1,2,3}_*/05_LGB_compare.ipynb`, `06_FNN_compare.ipynb` |
| Figs. 7–12 (feature dropping, Life / Diabetes / Bank Churn) | `0{1,2,3}_*/05_LGB_compare.ipynb`, `06_FNN_compare.ipynb` |
| Figs. 13–14, Crime/MIC top-30 tables | `04_crime/`, `05_mic/` (`01_LR_XAI.ipynb`, `02_LR_ShapG.ipynb`) |
| Figs. 15–16, Tables 7–8 | `04_crime/03_LR_compare.ipynb`, `05_mic/03_LR_compare.ipynb` |

## Installation

```bash
conda create -n shapley-xai python=3.10
conda activate shapley-xai
pip install -r requirements.txt
```

ShapG is published on PyPI as [`shapG`](https://github.com/vectorsss/shapG). The notebooks use the `shapG.shapley` / `shapG.utils` API, which was removed in later releases, so `requirements.txt` pins `shapG==0.12.1`.

### FastSHAP

The FastSHAP scripts use the PyTorch implementation from the FastSHAP authors (`fastshap_torch`). It is not included here. Get it from the authors' repository, for example [iancovert/fastshap](https://github.com/iancovert/fastshap), and make `fastshap_torch` importable. Then run:

```bash
cd fastshap
python fastshap_lgb.py diabetes   # or: Churn
python fastshap_fnn.py diabetes   # or: Churn
```

`fastshap/plot_fastshap.ipynb` plots the aggregated FastSHAP values.

## Data sources

- Life Expectancy (WHO): [Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who). `life.csv` is the preprocessed version and `normalized_life.csv` its min-max scaled version used for the FNN.
- Diabetes: dataset from a Chinese research study (see the paper). `normalized_diabetes.csv` is used for the FNN.
- Bank Churn: [Kaggle Playground Series S4E1](https://www.kaggle.com/competitions/playground-series-s4e1). `normalized_bank.csv` is used for the FNN.
- Communities and Crime: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/211/communities+and+crime+unnormalized). `crimedata_processing.csv` is created by `04_crime/00_data_processing.ipynb`.
- Myocardial Infarction Complications: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications). `MIC_processing.csv` is created by `05_mic/00_data_processing.ipynb`.

Datasets are redistributed for reproducibility only; please refer to the original sources for their licenses.

## Notes

- Running times were measured on the authors' machine and depend on hardware.
- KernelSHAP and SamplingSHAP were not run on Bank Churn because of their estimated running time (see the paper).

## Citation

```bibtex
@article{zhao_shapley_xai_survey,
  title  = {The Shapley value contribution to Explainable Artificial Intelligence: a comprehensive survey},
  author = {Zhao, Chi and Liu, Jing and Parilina, Elena},
}
```
