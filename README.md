# Healthcare Analytics for Doctor Visits_TIRTC

Mini project: analysing which factors drive doctor visits and predicting who is likely to visit.

## Dataset
`data/doctor_visits.csv` is the Australian Health Survey 1977-78 (DoctorVisits data): 5,190 people, 12 variables, no missing values.

| Column | Meaning |
|---|---|
| visits | Doctor visits in past 2 weeks (0-9) |
| gender, age, income | Sex; age in years/100; annual income in AUD 10,000s |
| illness | Illnesses in past 2 weeks (0-5) |
| reduced | Days of reduced activity in past 2 weeks |
| health | General health questionnaire score (higher = worse) |
| private, freepoor, freerepat | Private insurance; free government cover (low income / veterans) |
| nchronic, lchronic | Chronic condition without / with activity limitation |

## Approach
1. Encode gender and yes/no columns as 0/1.
2. Explore visit rates by age, income, illness, insurance and chronic condition.
3. Predict whether a person visited a doctor (logistic regression, random forest; 75/25 split).
4. Poisson regression for the expected number of visits.

## Results
| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 72.0% | 0.73 | 0.46 |
| Random Forest | 73.6% | 0.74 | 0.47 |

Strongest predictors: days of reduced activity and number of recent illnesses.

## Run
```bash
pip install -r requirements.txt
python analysis.py
```
Outputs (metrics JSON and charts) are written to `outputs/`. The submission slides are in `presentation/`.
