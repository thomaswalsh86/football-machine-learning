import pandas as pd
import pickle
import os

data_dir = "data"

# Load match results
with open(os.path.join(data_dir, "match_results.pkl"), 'rb') as f:
    matches = pickle.load(f)

print("Match Results Sample:")
print(matches.head())
print(f"Shape: {matches.shape}")
print("Columns:", matches.columns.tolist())

# Load teams
with open(os.path.join(data_dir, "teams.pkl"), 'rb') as f:
    teams = pickle.load(f)

print("\nTeams Sample:")
print(teams.head())
print(f"Shape: {teams.shape}")

# Check for Elo
elo_path = os.path.join(data_dir, "elo_ratings.csv")
if os.path.exists(elo_path):
    elo = pd.read_csv(elo_path)
    print("\nElo Ratings Sample:")
    print(elo.head())
    print(f"Shape: {elo.shape}")
else:
    print("\nElo ratings not found.")