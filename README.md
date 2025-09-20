# Flight Delay Prediction Hackathon

## Project Overview
Predict flight delays using airline and flight data. This project solves the regression problem of estimating total flight delay in minutes.

## Dataset
- Airline_Delay_Cause.csv: Contains flight delay cause details including carrier, weather, NAS, and other delay reasons.
- weather_data.csv: Weather information aligned with flights.
- Columns include: carrier, airport, delay types, arrival delay indicators, cancellation, diversion, etc.

## Feature Engineering
- Categorical Encoding: Carrier, Airport, Carrier-Airport interaction.
- Numeric Features: Arrival delay, cancellation flags, diversion flags.
- Target variable derived by summing individual delay types.

## Modelling Approach
- Baseline models: Random Forest, XGBoost.
- Hyperparameter tuning using GridSearchCV on XGBoost to optimize parameters.
- Model stacking with Random Forest and XGBoost predictions combined with linear regression.

## Evaluation
- Mean Absolute Error (MAE) as primary metric.
- Tuned XGBoost MAE achieved ~180 minutes or better.
- Visualizations include feature importance, predicted vs actual, and residuals histogram.

## Next Steps
- Further feature engineering including time features and external data.
- Model interpretability using SHAP.
- Ensemble expansion with other models.
- Final submission and deployment pipeline.

## Usage
To run training and evaluation:

