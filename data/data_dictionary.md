# Data dictionary: APIS 2022 integrated cleaned dataset (ages 5–24)

**Source:** Philippine Statistics Authority (PSA), Annual Poverty Indicators Survey 2022
(https://psada.psa.gov.ph/catalog/APIS/about). Member, Household and Other Relevant Information records were
merged by Region and Household ID and limited to ages 5–24 (72,527 rows; 72,194 after removing 333 exact duplicates).
The CSV (22 columns) is **not** included in this repository. Place `APIS_PUF_2022_Integrated_Cleaned.csv` in `data/`
before running `train_models.py`. To obtain the source files, request the APIS 2022 public-use files from PSA (link above).

| Column | Type | Role | Meaning | Known at prediction time? |
|---|---|---|---|---|
| school_attendance | binary | **Target** | 1 = attending (public, private or home-schooled), 0 = not attending | N/A |
| Household ID | int | Grouping only | Used so household members stay in the same split; never a predictor | N/A |
| Region | categorical | Predictor | Region of residence | Yes |
| Relationship to Household Head | categorical | Predictor | Person's relationship to the head | Yes |
| Sex | categorical | Predictor | Sex of the person | Yes |
| Age as of Last Birthday | numeric | Predictor | Age in years (5–24) | Yes |
| Marital Status | categorical | Predictor | Marital status of the person | Yes |
| Highest Grade Completed - Level | categorical | Predictor | Education level completed | Yes |
| Family Size | numeric | Predictor | Persons in the household | Yes |
| Head: Sex | categorical | Predictor | Household head's sex | Yes |
| Head: Age as of last birthday | numeric | Predictor | Household head's age | Yes |
| Head: Marital Status | categorical | Predictor | Household head's marital status | Yes |
| Head: Highest grade completed | categorical | Predictor | Household head's highest grade | Yes |
| Urban / Rural | categorical | Predictor | Residence classification | Yes |
| Average monthly income of the family January - June 2022 | numeric (PHP) | Predictor | Average monthly family income | Yes |

**Excluded to prevent leakage:** `Currently Attending School` (used to create the target), `Why not attending school`,
`Grade/Year currently attending - Code`, `Mode of learning` (all describe current schooling), plus
`Highest Grade Completed` (detailed), `Final Family Weight` and `Age Group` (the last is only used for subgroup analysis).
