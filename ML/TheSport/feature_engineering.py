import pandas as pd
import pickle
import os

data_dir = "data"

# Load data
with open(os.path.join(data_dir, "match_results.pkl"), 'rb') as f:
    matches = pickle.load(f)

elo_history = pd.read_csv(os.path.join(data_dir, "elo_history.csv"))
elo_history['date'] = pd.to_datetime(elo_history['date'])

# Sort matches
matches['utcDate'] = pd.to_datetime(matches['utcDate'])
matches = matches.sort_values('utcDate').reset_index(drop=True)

# Function to get Elo before match
def get_elo_before(team, date):
    team_elo = elo_history[(elo_history['home_team'] == team) | (elo_history['away_team'] == team)]
    team_elo = team_elo[team_elo['date'] < date]
    if not team_elo.empty:
        if team == team_elo.iloc[-1]['home_team']:
            return team_elo.iloc[-1]['home_elo']
        else:
            return team_elo.iloc[-1]['away_elo']
    return 1500  # Base

# Add Elo to matches
matches['home_elo'] = matches.apply(lambda x: get_elo_before(x['homeTeam.name'], x['utcDate']), axis=1)
matches['away_elo'] = matches.apply(lambda x: get_elo_before(x['awayTeam.name'], x['utcDate']), axis=1)
matches['elo_diff'] = matches['home_elo'] - matches['away_elo']

# Calculate team stats up to each match
team_stats = {}

def update_team_stats(team, goals_for, goals_against, result):
    if team not in team_stats:
        team_stats[team] = {'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0}
    stats = team_stats[team]
    stats['matches'] += 1
    stats['gf'] += goals_for
    stats['ga'] += goals_against
    if result == 'win':
        stats['wins'] += 1
    elif result == 'draw':
        stats['draws'] += 1
    else:
        stats['losses'] += 1

# Process matches to build stats
for idx, match in matches.iterrows():
    home = match['homeTeam.name']
    away = match['awayTeam.name']
    home_gf = match['score.fullTime.home'] or 0
    away_gf = match['score.fullTime.away'] or 0

    # Get current stats before this match
    home_stats = team_stats.get(home, {'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0})
    away_stats = team_stats.get(away, {'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0})

    matches.at[idx, 'home_matches'] = home_stats['matches']
    matches.at[idx, 'home_wins'] = home_stats['wins']
    matches.at[idx, 'home_draws'] = home_stats['draws']
    matches.at[idx, 'home_losses'] = home_stats['losses']
    matches.at[idx, 'home_gf'] = home_stats['gf']
    matches.at[idx, 'home_ga'] = home_stats['ga']

    matches.at[idx, 'away_matches'] = away_stats['matches']
    matches.at[idx, 'away_wins'] = away_stats['wins']
    matches.at[idx, 'away_draws'] = away_stats['draws']
    matches.at[idx, 'away_losses'] = away_stats['losses']
    matches.at[idx, 'away_gf'] = away_stats['gf']
    matches.at[idx, 'away_ga'] = away_stats['ga']

    # Update stats after match
    if home_gf > away_gf:
        update_team_stats(home, home_gf, away_gf, 'win')
        update_team_stats(away, away_gf, home_gf, 'loss')
    elif away_gf > home_gf:
        update_team_stats(home, home_gf, away_gf, 'loss')
        update_team_stats(away, away_gf, home_gf, 'win')
    else:
        update_team_stats(home, home_gf, away_gf, 'draw')
        update_team_stats(away, away_gf, home_gf, 'draw')

# Save features
features = matches[['id', 'utcDate', 'homeTeam.name', 'awayTeam.name', 'home_elo', 'away_elo', 'elo_diff',
                    'home_matches', 'home_wins', 'home_draws', 'home_losses', 'home_gf', 'home_ga',
                    'away_matches', 'away_wins', 'away_draws', 'away_losses', 'away_gf', 'away_ga',
                    'score.fullTime.home', 'score.fullTime.away']]
features.to_csv(os.path.join(data_dir, "match_features.csv"), index=False)
print("Features saved.")