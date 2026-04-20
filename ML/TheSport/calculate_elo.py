import pandas as pd
import pickle
import os
import numpy as np

data_dir = "data"

# Load match results
with open(os.path.join(data_dir, "match_results.pkl"), 'rb') as f:
    matches = pickle.load(f)

# Sort matches by date
matches['utcDate'] = pd.to_datetime(matches['utcDate'])
matches = matches.sort_values('utcDate').reset_index(drop=True)

# Initialize Elo ratings
elo_ratings = {}
base_elo = 1500
k_factor = 30

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(winner_elo, loser_elo, draw=False):
    exp_w = expected_score(winner_elo, loser_elo)
    exp_l = 1 - exp_w
    if draw:
        new_w = winner_elo + k_factor * (0.5 - exp_w)
        new_l = loser_elo + k_factor * (0.5 - exp_l)
    else:
        new_w = winner_elo + k_factor * (1 - exp_w)
        new_l = loser_elo + k_factor * (0 - exp_l)
    return new_w, new_l

# Get unique teams
home_teams = matches['homeTeam.name'].unique()
away_teams = matches['awayTeam.name'].unique()
all_teams = set(home_teams) | set(away_teams)
for team in all_teams:
    elo_ratings[team] = base_elo

# Elo history
elo_history = []

# Process each match
for idx, match in matches.iterrows():
    home = match['homeTeam.name']
    away = match['awayTeam.name']
    home_score = match['score.fullTime.home']
    away_score = match['score.fullTime.away']

    home_elo = elo_ratings[home]
    away_elo = elo_ratings[away]

    # Record before update
    elo_history.append({
        'match_id': match['id'],
        'date': match['utcDate'],
        'home_team': home,
        'away_team': away,
        'home_elo': home_elo,
        'away_elo': away_elo
    })

    if home_score > away_score:
        # Home win
        new_home, new_away = update_elo(home_elo, away_elo)
    elif away_score > home_score:
        # Away win
        new_away, new_home = update_elo(away_elo, home_elo)
    else:
        # Draw
        new_home, new_away = update_elo(home_elo, away_elo, draw=True)

    elo_ratings[home] = new_home
    elo_ratings[away] = new_away

# Save Elo history
elo_df = pd.DataFrame(elo_history)
elo_df.to_csv(os.path.join(data_dir, "elo_history.csv"), index=False)
print("Elo history saved.")

# Current Elo
current_elo = pd.DataFrame(list(elo_ratings.items()), columns=['team', 'elo'])
current_elo.to_csv(os.path.join(data_dir, "current_elo.csv"), index=False)
print("Current Elo saved.")