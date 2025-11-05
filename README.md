Python Machine Learning Case Study: Ames Housing Price Prediction
Project Overview
This project demonstrates a complete machine learning workflow for predicting home sale prices using the Ames Housing dataset. The analysis follows best practices in data science, including exploratory data analysis, feature engineering, time-based train-test splitting, and model evaluation with interpretable error analysis.
Dataset

Source: AmesHousing.csv
Target Variable: SalePrice (predicting sale price of single-family homes)
Records: 2,929 home sales
Features: 70 columns including property characteristics, location, and sale details

Project Structure
1. Data Quality & Initial Exploration

Loaded the Ames Housing dataset with no duplicate records
Examined data types and missing values using .info() and .describe()
Visualized SalePrice distribution (showing right-skew typical for housing prices)

2. Feature Engineering
Created a Total Area feature combining:

1st Floor SF
2nd Floor SF
Total Basement SF

This provides a compact proxy for overall home size.
3. Feature Screening

Computed numeric correlations with SalePrice
Generated correlation heatmap to identify strong relationships
Key findings:

Total Area: 0.793 correlation (strongest predictor)
Overall Quality: 0.799 correlation
Year Built: 0.558 correlation



Used regression plots to check linearity and outliers, confirming Total Area shows clear positive trend.
4. Train/Test Strategy
Implemented time-based split to prevent data leakage:

Training Set: Yr Sold < 2009 (1,940 records)
Test Set: Yr Sold ≥ 2009 (989 records)

This mimics real-world future prediction scenarios.
5. Baseline Model
Built simple Linear Regression with two features:

Total Area
Year Built

Rationale: Establish transparent, explainable starting point before adding complexity.
6. Model Evaluation
Overall Performance

Mean Absolute Error (MAE): $27,114
Average prediction error of ~$27K on test set

Error Analysis by Neighborhood
Computed absolute errors per sale and grouped by neighborhood:
Best Performing Neighborhoods (lowest relative error):

CollgCr: 8.8%
Somerst: 8.8%
Gilbert: 9.0%

Worst Performing Neighborhoods (highest relative error):

MeadowV: 34.1%
OldTown: 24.4%
Edwards: 24.4%

Insights: Model underperforms in lower-priced neighborhoods (MeadowV), suggesting need for additional features beyond just size and age.
7. Error Distribution
Histogram of relative errors shows most neighborhoods fall in 10-20% range, with long tail indicating some areas need improved modeling.
Key Takeaways
What Works
✅ Total Area is strong predictor (79% correlation)
✅ Time-based split prevents leakage
✅ Baseline MAE of $27K establishes benchmark
✅ Error analysis reveals geographic patterns
Next Steps for Improvement
🔄 Log-transform SalePrice to handle right-skew
🔄 Add quality/condition features (Overall Qual shows 80% correlation)
🔄 Include neighborhood-specific features
🔄 Apply regularization (Ridge/Lasso) to reduce overfitting
🔄 Test polynomial features for non-linear relationships
Technologies Used

Python 3.x
pandas: Data manipulation
matplotlib/seaborn: Visualization
scikit-learn: Machine learning (LinearRegression, train_test_split, metrics)

How to Run
python# Load data
import pandas as pd
df = pd.read_csv("AmesHousing.csv")

# Create features
df["Total Area"] = df["1st Flr SF"] + df["2nd Flr SF"] + df["Total Bsmt SF"]

# Time-based split
train = df[df["Yr Sold"] < 2009]
test = df[df["Yr Sold"] >= 2009]

# Train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train[["Total Area", "Year Built"]], train["SalePrice"])

# Evaluate
predictions = model.predict(test[["Total Area", "Year Built"]])
Results Summary
This baseline model achieves reasonable performance (~15% relative error on average) using only two interpretable features. The neighborhood-level error analysis provides actionable insights for targeted model improvements, particularly for lower-priced areas where simple size/age metrics are insufficient.
Author
Machine Learning Case Study - Hex Project
License
Educational/Research Use

Note: This analysis prioritizes interpretability and reproducibility over raw performance. The simple two-feature model establishes a transparent baseline for comparison with more complex approaches.
