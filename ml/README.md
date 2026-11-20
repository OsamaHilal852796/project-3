# SPWS - Machine Learning part

| File | Purpose |
|---|---|
| `ridge_model.py` | The model: `ridge_train`, `ridge_predict`, `predicted_total`, `risk`, `predict_student` |
| `train_and_evaluate.py` | Evaluation: MAE, RMSE, R2, baseline, confusion matrix; supports hold-out split |
| `seed.json` | Stored models + predictions for the two subjects (729 students each) |

Algorithm: Ridge regression (L2), closed form, z-score standardisation, 7 input features.

```bash
pip install -r requirements.txt
python train_and_evaluate.py seed.json                       # reproduces the report numbers
python train_and_evaluate.py students.xlsx --holdout 0.2     # honest evaluation on unseen students
```
`students.xlsx` needs the columns: quiz1_lec, quiz2_lec, total_lecture, quiz1_section, quiz2_section, total_section, midterm, final, final_total.

## Run on the dataset
```bash
python run_on_dataset.py                 # uses dataset_spws.csv (1,458 students, names/IDs removed)
```
`dataset_spws.csv` holds the columns stored in the system: subject, midterm, total_lecture, total_section, final, final_total, passed.
The quiz columns were not stored, so this run uses 3 features; with the original spreadsheet (7 features) pass its path as an argument.
