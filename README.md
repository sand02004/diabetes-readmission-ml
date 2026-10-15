# Predicting 30-Day Hospital Readmission in Diabetic Patients

Traditional machine learning vs deep learning on the UCI Diabetes 130-US Hospitals dataset (101,766 encounters, 3 classes: `NO`, `>30`, `<30`).

**Author:** Sandrine Umugwaneza, African Leadership University

## Links
- **Report:** [add link]
- **Google Drive** (raw dataset zip, best models, logs): [add link]
- **Video presentation:** [add link]

## Problem
Can we predict whether a diabetic patient will be readmitted within 30 days, after 30 days, or not at all, using data from the hospital stay? The `<30` class is the clinically critical one and the rarest (11%).

## Dataset
Strack et al. (2014), UCI Machine Learning Repository, DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J), licensed CC BY 4.0.

**The raw data is not in this repository.** Download `diabetic_data.csv` and `IDS_mapping.csv` from the Google Drive link above (or from UCI), then set `DATA_PATH` in the notebook.

## Approach
- Cleaning: dropped `weight` (97% missing), treated untested labs as "Not tested", removed patients who died or went to hospice.
- Feature engineering: age midpoints, ICD-9 diagnosis groups, total prior visits, medication counts, grouped rare specialties.
- Patient-level 70/15/15 split (no patient appears in two sets).
- 11 experiments: Logistic Regression, Random Forest, Histogram Gradient Boosting, Keras MLP (Sequential API with a `tf.data` pipeline), regularisation and learning-rate tuning, threshold tuning, and a feature ablation.
- Evaluation: macro-F1, per-class recall and precision, one-vs-rest ROC-AUC, confusion matrices, learning curves, permutation importance.
- Experiment tracking: Weights & Biases in offline mode.

## Final results (held-out test set)
| Model | Accuracy | Macro-F1 | Macro AUC | `<30` recall | `<30` precision |
|---|---|---|---|---|---|
| Gradient Boosting + threshold (E10) | 0.558 | 0.468 | 0.674 | 0.240 | 0.276 |
| Keras MLP (E5) | 0.531 | 0.427 | 0.674 | 0.389 | 0.220 |

All model families reach a similar AUC (about 0.67), so the features, not the algorithm, limit performance.

## Repository contents
- `readmission_prediction.ipynb`: full notebook with markdown explanations and interpretations
- `requirements.txt`: dependencies
- `.gitignore`: excludes data, models and logs

## Run it
1. `pip install -r requirements.txt`
2. Put `diabetic_data.csv` in a folder and set `DATA_PATH` in the notebook (written for Google Colab with Drive; change the mount cell if you run locally).
3. Run all cells top to bottom.

Tested on Python 3.13 (Google Colab). Results may vary slightly (about ±0.005) across environments because of library versions and TensorFlow numerics. The values reported in the paper come from the run archived in `wandb_logs.zip` on the Drive folder.

## Reproducibility
Random seed 42 is used everywhere. Experiment logs for all 11 runs are in `wandb_logs.zip` (Drive), matching the experiment table in the notebook.

## Reference
B. Strack et al., "Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records," *BioMed Research International*, vol. 2014, Art. no. 781670, 2014.
