# 🏡 Ames Housing Price Prediction

A machine learning exploration into understanding what drives home prices in Ames, Iowa. This project walks through the complete data science journey—from initial data exploration to building and evaluating a predictive model.

## 📖 Project Story

What makes a house valuable? Is it the size, the age, the neighborhood, or something else entirely? This project started with a simple question: **Can we predict home sale prices using basic property characteristics?**

Using the Ames Housing dataset, I embarked on an exploratory data analysis (EDA) journey to uncover patterns in residential real estate pricing. Through systematic feature engineering, correlation analysis, and model building, this case study demonstrates how even simple machine learning models can reveal meaningful insights about housing markets—and where they fall short.

The goal wasn't just to achieve high accuracy, but to build an **interpretable, transparent baseline** that could tell us *why* certain predictions work better than others.

## 📊 Dataset

- **Source:** AmesHousing.csv
- **Target Variable:** `SalePrice` (single-family home sale prices)
- **Records:** 2,929 home sales
- **Features:** 70 columns including property characteristics, location details, and sale information
- **Time Period:** Sales from 2006-2010

## 🔍 Methodology

### 1. **Data Quality & Exploration**
- Loaded and validated dataset integrity (no duplicates found)
- Examined data types, missing values, and summary statistics
- Visualized `SalePrice` distribution (revealed right-skewed pattern typical of real estate)

<img width="613" height="373" alt="image" src="https://github.com/user-attachments/assets/22fbae9f-337d-45e1-91a0-e595d65f0754" />

*The right-skewed distribution shows most homes cluster in the $100K-$200K range, with a long tail of luxury properties.*

### 2. **Feature Engineering**
Created a **Total Area** feature combining:
- 1st Floor SF
- 2nd Floor SF  
- Total Basement SF

This composite metric serves as a comprehensive proxy for overall home size.

### 3. **Feature Screening**
- Computed correlation matrix for all numeric features
- Generated heatmap visualization to identify relationships
- **Key Discoveries:**
  - `Total Area`: **0.793 correlation** (strongest size predictor)
  - `Overall Qual`: **0.799 correlation** (quality matters!)
  - `Year Built`: **0.558 correlation** (newer = pricier)

<img width="655" height="662" alt="image" src="https://github.com/user-attachments/assets/9ace65dc-a0c2-45dc-b0f1-88e150949fbb" />

*Correlation heatmap revealing the strongest predictors of sale price.*

Used regression plots to validate linear relationships and detect outliers.

<img width="607" height="380" alt="image" src="https://github.com/user-attachments/assets/0c2b255a-67a2-4567-81c3-eca3b1f5a527" />

*Strong linear relationship between Total Area and SalePrice (r=0.793), with a few notable outliers.*

### 4. **Train/Test Strategy**
Implemented **time-based split** to simulate real-world forecasting:
- **Training Set:** Sales before 2009 (1,940 records)
- **Test Set:** Sales in 2009+ (989 records)

This prevents data leakage and mirrors actual prediction scenarios where you forecast future prices using historical data.

### 5. **Baseline Model**
Built a simple **Linear Regression** model with two features:
- `Total Area`
- `Year Built`

**Rationale:** Establish an explainable starting point before adding complexity. Sometimes simple is better for understanding cause and effect.

## 📈 Results

### Overall Performance
- **Mean Absolute Error (MAE):** $27,114
- **Average Relative Error:** ~15%

### Neighborhood-Level Insights

**Best Performing Neighborhoods** (lowest error):
| Neighborhood | Relative Error |
|--------------|---------------|
| CollgCr      | 8.8%          |
| Somerst      | 8.8%          |
| Gilbert      | 9.0%          |

**Worst Performing Neighborhoods** (highest error):
| Neighborhood | Relative Error |
|--------------|---------------|
| MeadowV      | 34.1%         |
| OldTown      | 24.4%         |
| Edwards      | 24.4%         |

**Key Finding:** The model struggles in lower-priced neighborhoods like MeadowV, suggesting that size and age alone don't capture what makes these homes valuable (or not). Additional features like condition, amenities, or micro-location factors are likely needed.

### Error Distribution
Most neighborhoods cluster in the 10-20% error range, but a long tail indicates specific areas where the simple model breaks down.

## 🎯 Key Takeaways

### What Works ✅
- Total Area proves to be a strong, interpretable predictor (79% correlation)
- Time-based splitting ensures realistic evaluation
- $27K MAE establishes a solid baseline for comparison
- Neighborhood error analysis reveals geographic patterns worth investigating

### What We Learned 💡
- Simple features like size and age capture broad trends but miss neighborhood-specific nuances
- Lower-priced areas require different modeling approaches
- Quality ratings (`Overall Qual` at 80% correlation) should be prioritized in next iterations
- The right-skewed price distribution suggests logarithmic transformation could help

## 🚀 Next Steps

This project establishes a foundation for more sophisticated approaches. Future improvements to explore:

1. **Feature Enhancement**
   - Log-transform `SalePrice` to handle right-skew distribution
   - Incorporate `Overall Qual` and `Overall Cond` ratings
   - Add categorical features (neighborhood dummies, property style)
   - Engineer interaction terms (e.g., Area × Quality)

2. **Model Development**
   - Test Ridge/Lasso regression to reduce overfitting
   - Experiment with polynomial features for non-linear relationships
   - Try ensemble methods (Random Forest, Gradient Boosting)
   - Implement cross-validation for more robust evaluation

3. **Deep-Dive Analysis**
   - Investigate MeadowV and other underperforming neighborhoods
   - Analyze prediction residuals for systematic patterns
   - Build neighborhood-specific models
   - Create feature importance rankings

4. **Deployment Considerations**
   - Package model for API deployment
   - Build simple web interface for price predictions
   - Add confidence intervals to predictions
   - Monitor model drift over time

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **matplotlib/seaborn** - Data visualization
- **scikit-learn** - Machine learning (LinearRegression, metrics)
- **NumPy** - Numerical computing

## 💻 Quick Start

```python
# Load data
import pandas as pd
df = pd.read_csv("AmesHousing.csv")

# Create Total Area feature
df["Total Area"] = df["1st Flr SF"] + df["2nd Flr SF"] + df["Total Bsmt SF"]

# Time-based split
train = df[df["Yr Sold"] < 2009]
test = df[df["Yr Sold"] >= 2009]

# Train baseline model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train[["Total Area", "Year Built"]], train["SalePrice"])

# Generate predictions
predictions = model.predict(test[["Total Area", "Year Built"]])

# Evaluate
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test["SalePrice"], predictions)
print(f"Mean Absolute Error: ${mae:,.0f}")
```

## 📁 Project Structure

```
ames-housing-prediction/
│
├── data/
│   └── AmesHousing.csv
│
├── notebooks/
│   └── ames_analysis.ipynb
│
├── visualizations/
│   ├── saleprice_distribution.png
│   ├── correlation_heatmap.png
│   ├── total_area_regression.png
│   └── error_distribution.png
│
└── README.md
```

## 🎓 Learning Outcomes

This project demonstrates:
- Proper train/test splitting for time-series data
- Feature engineering and correlation analysis
- Baseline model establishment
- Interpretable error analysis by segment
- The importance of starting simple before adding complexity


Machine Learning Case Study
---

