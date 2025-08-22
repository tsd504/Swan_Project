# Swan Project - Customer Churn Prediction

This project focuses on predicting customer churn for Swan, a telecommunications company. Customer churn is a critical business metric that directly impacts revenue and growth. The goal is to identify which customers are most likely to leave the service, enabling proactive retention strategies and reducing customer loss.

## Project Structure

```
Swan_Project/
├── 1. Jagroop/           # EDA and Random Forest
│   ├── EDA.ipynb         # Exploratory data analysis
│   ├── FE.ipynb          # Feature engineering
│   ├── Modelling.ipynb   # Machine learning models
│   └── Swan_Churn.csv    # Customer dataset
├── 2. Saad/              # Advanced analysis and visualisation
│   ├── EDA.ipynb         # Comprehensive EDA with visualisations
│   ├── FE.ipynb          # Feature engineering techniques
│   ├── Modelling.ipynb   # Model development
│   ├── Pretty copy.ipynb # Final presentation notebook
│   ├── ydf_swan_project.ipynb # Yellowbrick diagnostics
│   └── Project Data.csv  # Customer dataset
├── 3. Tom/               # Final modelling and insights
│   ├── EDA.ipynb         # Detailed exploratory analysis
│   ├── FE.ipynb          # Feature engineering
│   ├── Modelling.ipynb   # Model training and evaluation
│   ├── Final_notebook.ipynb # Complete analysis pipeline
│   ├── data.csv          # Customer dataset
│   ├── remaining_customers.csv # Non-churned customers
│   └── top_500.csv       # High-risk customers
└── Project Brief.pdf     # Project requirements and objectives
```

## Dataset Overview

The project utilises a comprehensive customer dataset containing **7,043 customers** with the following key information:

### Customer Demographics
- **Gender**: Male/Female
- **Senior Citizen**: Yes/No
- **Partner**: Yes/No (relationship status)
- **Dependents**: Yes/No

### Service Information
- **Phone Service**: Yes/No
- **Multiple Lines**: Yes/No/No phone service
- **Internet Service**: DSL/Fiber optic/No
- **Online Security**: Yes/No/No internet service
- **Online Backup**: Yes/No/No internet service
- **Device Protection**: Yes/No/No internet service
- **Tech Support**: Yes/No/No internet service
- **Streaming TV**: Yes/No/No internet service
- **Streaming Movies**: Yes/No/No internet service

### Contract Details
- **Contract**: Month-to-month/One year/Two year
- **Paperless Billing**: Yes/No
- **Payment Method**: Mailed check/Electronic check/Bank transfer/Credit card
- **Monthly Charges**: Continuous variable
- **Total Charges**: Continuous variable

### Target Variable
- **Churn Value**: Binary (0 = No churn, 1 = Churned) - **Historical outcome used for training**
- **Churn Label**: Categorical (Yes/No) - **Alternative representation of churn status**
- **Churn Reason**: Text description for churned customers

**Note**: We use historical churn data to predict the probability of future churning. The model outputs a churn risk score (0-1) indicating likelihood of leaving.

## Key Findings

### Data Characteristics
- **Class Imbalance**: 26.5% churn rate (1,869 churned customers)
- **Geographic Focus**: All customers located in California, United States
- **Service Patterns**: Strong correlation between internet service and additional features
- **Contract Impact**: Month-to-month contracts show higher churn rates

### Important Features
1. **Contract Type**: Longer contracts (2-year) have significantly lower churn rates
2. **Internet Service**: Fiber optic customers churn more than DSL customers
3. **Monthly Charges**: Higher charges correlate with increased churn risk
4. **Tenure**: Newer customers (0-12 months) are more likely to churn
5. **Payment Method**: Electronic checks associated with higher churn

## Methodology

### 1. Exploratory Data Analysis (EDA)
- **Data Quality Assessment**: Identified missing values, duplicates, and inconsistencies
- **Feature Analysis**: Explored relationships between variables and churn
- **Visualisation**: Created comprehensive charts and heatmaps
- **Statistical Insights**: Analysed correlations and distributions

### 2. Feature Engineering
- **Categorical Encoding**: Converted text variables to numerical representations
- **Feature Creation**: Developed meaningful combinations of existing features
- **Data Cleaning**: Handled missing values and standardised formats
- **Multicollinearity Analysis**: Identified and addressed redundant features

### 3. Machine Learning Models
- **Logistic Regression**: Primary model with best overall performance
- **Decision Trees**: Baseline comparison model
- **Random Forest**: Ensemble method for comparison
- **Cross-validation**: Ensured robust model evaluation
- **SMOTE**: Addressed class imbalance issues

### 4. Model Evaluation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Detailed classification results
- **ROC Curves**: Model discrimination ability
- **Feature Importance**: Understanding key predictors

## Results and Insights

### Model Performance
- **Best Model**: Logistic Regression with SMOTE balancing
- **Accuracy**: 74.9% on test set with balanced classes
- **Business Impact**: Identified high-risk customers for targeted retention

### Key Recommendations
1. **Contract Incentives**: Encourage longer-term contracts
2. **Service Bundling**: Package internet services with additional features
3. **Early Intervention**: Focus on customers in first 12 months
4. **Payment Options**: Promote automatic payment methods
5. **Customer Support**: Enhance technical support for fiber optic customers


## Technical Implementation

### Technologies Used
- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualisation
- **SMOTE**: Class balancing technique

### Key Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
```



## Future Enhancements

### Model Improvements
- **Deep Learning**: Neural networks for complex pattern recognition
- **Ensemble Methods**: Stacking and blending multiple models


### Business Applications
- **Real-time Scoring**: Live churn risk assessment
- **Customer Segmentation**: Behavioural clustering analysis
- **Retention Campaigns**: Targeted marketing strategies


---

*This project showcases the complete data science pipeline from exploratory analysis to actionable business insights, demonstrating how machine learning can drive customer retention strategies.*
