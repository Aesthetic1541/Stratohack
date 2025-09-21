from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

from utils import load_and_clean_data, encode_categorical_columns, create_interaction_feature, save_feature_importance_plot

DATA_PATH = './flight-delay-prediction/data/Airline_Delay_Cause.csv'

# Load and clean data
df = load_and_clean_data(DATA_PATH)

# Encode categorical variables
df = encode_categorical_columns(df, ['carrier', 'airport'])

# Create interaction feature
df = create_interaction_feature(df, 'carrier', 'airport', 'carrier_airport_code')

# Define features and target column
features = [
    'month',
    'carrier_code',
    'airport_code',
    'carrier_airport_code',
    'arr_delay',
    'arr_del15',
    'arr_cancelled',
    'arr_diverted'
]

df['Total_Delay'] = df[['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']].sum(axis=1)

X = df[features]
y = df['Total_Delay']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search parameters
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.05],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

grid_search = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("Test Mean Absolute Error:", mae)

# Save feature importance plot
save_feature_importance_plot(best_model, features, 'feature_importances.png')

# Save predictions to CSV
results = X_test.copy()
results['predicted_delay'] = y_pred
results.to_csv('prediction_results.csv', index=False)
print("Predictions saved to prediction_results.csv")
