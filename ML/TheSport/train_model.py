import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import os

data_dir = "data"

# Load features
features = pd.read_csv(os.path.join(data_dir, "match_features.csv"))

# Drop rows with missing feature values (first few matches have no rolling stats)
features = features.dropna()

# Create target: 2 = home win, 1 = draw, 0 = away win
def get_target(row):
    if row['home_score'] > row['away_score']:
        return 2
    elif row['home_score'] == row['away_score']:
        return 1
    else:
        return 0

features['target'] = features.apply(get_target, axis=1)

# ---------------------------
# Season‑based train/test split (no data leakage)
# We assume the 'utcDate' column is present and we extract the year
# or you can use a 'season' column if you have one from main.py
# ---------------------------
# If you added a 'season' column in feature_engineering.py, use it directly.
# Here we'll extract season from the date (the date format is like "2017-08-11")
features['season_year'] = pd.to_datetime(features['utcDate']).dt.year

# Define training seasons: all seasons before the last one
# Our data spans from 2017/18 to 2024/25, so the last season starts in 2024.
# We train on matches before 2024 and test on matches from 2024 onwards.
train = features[features['season_year'] < 2024]
test = features[features['season_year'] >= 2024]

print(f"Training set size: {len(train)}")
print(f"Test set size: {len(test)}")

# Feature columns – must match the columns produced by feature_engineering.py
feature_cols = [
    'home_elo', 'away_elo', 'elo_diff',
    'home_points_last5', 'away_points_last5',
    'home_xg_last5', 'away_xg_last5', 'home_xga_last5', 'away_xga_last5',
    'home_possession_last5', 'away_possession_last5',
    'home_rest_days', 'away_rest_days', 'home_matches_7d', 'away_matches_7d',
    'home_form_ewma', 'away_form_ewma',
    'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_avg_goals'
]

X_train = train[feature_cols]
y_train = train['target']
X_test = test[feature_cols]
y_test = test['target']

# ---------------------------
# Train XGBoost classifier
# ---------------------------
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# ---------------------------
# Evaluate
# ---------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))

# Save the model
model.save_model(os.path.join(data_dir, "xgb_model.json"))
print("\nModel saved to xgb_model.json")