# Customer Churn Prediction & Lifetime Value (LTV) Engine

##  Project Overview
A predictive analytics system designed for telecommunications businesses.
This engine identifies customers at high risk of cancellation (churn) and
calculates Customer Lifetime Value (LTV) to help marketing teams prioritize
high-value retention campaigns.

##  Team Members
- R Dhanalakshmi (Team Leader)
- Ch Mounya
- N Sathwika
- L Divya
- Sindhu


##  Dataset
- **Source:** Telco Customer Churn Dataset (Kaggle)
- **Size:** 7,043 customers, 21 features
- **Churn Rate:** 26.54%

##  Key Insights
1. Senior citizens churn 41.7% vs 23.6% non-seniors
2. Electronic check users churn 45.3%
3. Fiber optic users churn 41.9%
4. New customers (< 12 months) churn 48.3%
5. More services = less churn (7 services = 5.8% churn)
6. Month-to-month contracts churn the most

##  Tech Stack
- **Language:** Python, SQL
- **Database:** PostgreSQL
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Seaborn
- **API:** FastAPI
- **Visualization:** Matplotlib, Seaborn

##  Project Structure
customer-churn-ltv/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── customers_clean.csv
│   └── customers_featured.csv
├── models/
│   ├── churn_model.pkl
│   └── feature_names.pkl
├── notebooks/
│   ├── eda.ipynb
│   ├── database.ipynb
│   ├── eda_analysis.ipynb
│   ├── data_preprocessing.ipynb
│   ├── baseline_report.ipynb
│   └── feature_engineering.ipynb
└── README.md

## 📅 Weekly Progress
### Week 1 
- Day 1: EDA, visualizations, ML models, LTV calculation
- Day 2: PostgreSQL setup and dataset loading
- Day 3: Detailed EDA analysis
- Day 4: Data preprocessing
- Day 5: Baseline analytics report
- Day 6: Feature engineering
- Day 7: Final cleanup and documentation

### Week 2 
- Day 1-3: Feature engineering improvements
- Day 4-6: Improved ML models with GridSearchCV
- Day 7: SHAP values and best model saved

### Week 3 
- Day 1-3: LTV regression model built
- Day 4-7: FastAPI with /predict and /ltv endpoints

### Week 4 
- Day 1-3: Streamlit dashboard with 4 pages
- Day 4-5: Interactive visualizations and testing
- Day 6-7: Dockerfile and final documentation

## 📈 Model Performance
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 80.91% |
| Random Forest | 80.84% |
| XGBoost | 79.91% |
| LTV Model (Random Forest) | 99.93% R2 Score |

##  How to Run
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Setup PostgreSQL database
4. Run notebooks in order
