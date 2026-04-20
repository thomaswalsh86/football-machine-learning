import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap
import os
import matplotlib.pyplot as plt

data_dir = "data"
features = pd.read_csv(os.path.join(data_dir, "match_features.csv"))

# Drop rows with NaN (early matches with no stats)
features = features.dropna()

# Create target: 2 = home win, 1 = draw, 0 = away win
def get_target(row):
    if row['score.fullTime.home'] > row['score.fullTime.away']:
        return 2
    elif row['score.fullTime.home'] == row['score.fullTime.away']:
        return 1
    else:
        return 0

features['target'] = features.apply(get_target, axis=1)

# Features to use
feature_cols = ['home_elo', 'away_elo', 'elo_diff',
                'home_matches', 'home_wins', 'home_draws', 'home_losses', 'home_gf', 'home_ga',
                'away_matches', 'away_wins', 'away_draws', 'away_losses', 'away_gf', 'away_ga']

X = features[feature_cols]
y = features['target']

# Split: first 300 train, rest test
X_train = X[:300]
y_train = y[:300]
X_test = X[300:]
y_test = y[300:]

# XGBoost model
model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=100, max_depth=3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# SHAP - commented out for speed
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
# plt.savefig(os.path.join(data_dir, "shap_summary.png"))
# print("SHAP plot saved.")

# Save model
model.save_model(os.path.join(data_dir, "xgb_model.json"))
print("Model saved.")