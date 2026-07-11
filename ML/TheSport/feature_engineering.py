import pandas as pd
import pickle
import os
import numpy as np

data_dir = "data"

# Load matches (with xG, possession, etc.)
with open(os.path.join(data_dir, "match_results.pkl"), 'rb') as f:
    matches = pickle.load(f)

# Load Elo history
elo_history = pd.read_csv(os.path.join(data_dir, "elo_history.csv"))
elo_history['date'] = pd.to_datetime(elo_history['date']).dt.tz_localize(None)

# Sort matches chronologically
matches = matches.sort_values('utcDate').reset_index(drop=True)
matches['utcDate'] = pd.to_datetime(matches['utcDate']).dt.tz_localize(None)

# ----- Rolling stats containers -----
team_past_matches = {}  # key: team name, value: list of dicts with match stats
h2h_records = {}        # key: tuple (teamA, teamB) sorted, value: list of past results

def update_team_history(team, match_date, goals_for, goals_against, xg_for, xg_against, possession, result):
    if team not in team_past_matches:
        team_past_matches[team] = []
    team_past_matches[team].append({
        'date': match_date,
        'gf': goals_for,
        'ga': goals_against,
        'xg': xg_for,
        'xga': xg_against,
        'possession': possession,
        'result': result   # 'win', 'draw', 'loss'
    })

def update_h2h(home, away, match_date, home_score, away_score):
    key = tuple(sorted([home, away]))
    if key not in h2h_records:
        h2h_records[key] = []
    h2h_records[key].append({
        'date': match_date,
        'home_team': home,
        'away_team': away,
        'home_score': home_score,
        'away_score': away_score
    })

# ---- Elo retrieval function (original, safe) ----
def get_elo_before(team, date, elo_df):
    """Return the most recent Elo for a team before the given date."""
    team_elo = elo_df[(elo_df['home_team'] == team) | (elo_df['away_team'] == team)]
    team_elo = team_elo[team_elo['date'] < date]
    if not team_elo.empty:
        last_match = team_elo.iloc[-1]
        if team == last_match['home_team']:
            return last_match['home_elo']
        else:
            return last_match['away_elo']
    return 1500  # Base

# ---- Feature computation functions ----
def compute_rolling_stats(team, current_date):
    if team not in team_past_matches or len(team_past_matches[team]) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    history = team_past_matches[team]
    past = [m for m in history if m['date'] < current_date]
    if not past:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    last5 = sorted(past, key=lambda x: x['date'], reverse=True)[:5]
    points = sum(3 if m['result'] == 'win' else 1 if m['result'] == 'draw' else 0 for m in last5) / len(last5)
    xg = np.mean([m['xg'] for m in last5])
    xga = np.mean([m['xga'] for m in last5])
    poss = np.mean([m['possession'] for m in last5])
    last_match_date = max(m['date'] for m in past)
    rest_days = (current_date - last_match_date).days
    matches_7d = sum(1 for m in past if (current_date - m['date']).days <= 7)
    return (points, xg, xga, poss, rest_days, matches_7d)

def compute_form_ewma(team, current_date, decay=0.9):
    if team not in team_past_matches:
        return 0.5
    past = [m for m in team_past_matches[team] if m['date'] < current_date]
    if not past:
        return 0.5
    past_sorted = sorted(past, key=lambda x: x['date'])
    results = [1.0 if m['result']=='win' else 0.5 if m['result']=='draw' else 0.0 for m in past_sorted]
    weights = [decay**i for i in reversed(range(len(results)))]
    weight_sum = sum(weights)
    if weight_sum == 0:
        return 0.5
    return sum(r*w for r,w in zip(results, weights)) / weight_sum

def compute_h2h(home, away, current_date):
    key = tuple(sorted([home, away]))
    if key not in h2h_records:
        return (0.5, 0.5, np.nan)
    past = [m for m in h2h_records[key] if m['date'] < current_date]
    if not past:
        return (0.5, 0.5, np.nan)
    last4 = sorted(past, key=lambda x: x['date'], reverse=True)[:4]
    home_wins = sum(1 for m in last4 if m['home_team'] == home and m['home_score'] > m['away_score'])
    away_wins = sum(1 for m in last4 if m['away_team'] == away and m['away_score'] > m['home_score'])
    home_rate = home_wins / len(last4)
    away_rate = away_wins / len(last4)
    avg_goals = np.mean([m['home_score'] + m['away_score'] for m in last4])
    return (home_rate, away_rate, avg_goals)

# ---- Main processing loop ----
features_list = []
for idx, match in matches.iterrows():
    home = match['homeTeam.name']
    away = match['awayTeam.name']
    current_date = match['utcDate']

    # 1. Elo ratings (using the safe function)
    home_elo = get_elo_before(home, current_date, elo_history)
    away_elo = get_elo_before(away, current_date, elo_history)
    elo_diff = home_elo - away_elo

    # 2. Rolling performance features
    home_points5, home_xg5, home_xga5, home_poss5, home_rest, home_m7d = compute_rolling_stats(home, current_date)
    away_points5, away_xg5, away_xga5, away_poss5, away_rest, away_m7d = compute_rolling_stats(away, current_date)

    # 3. Form EWMA
    home_ewma = compute_form_ewma(home, current_date)
    away_ewma = compute_form_ewma(away, current_date)

    # 4. Head-to-head
    h2h_home_win_rate, h2h_away_win_rate, h2h_avg_goals = compute_h2h(home, away, current_date)

    # 5. Current match stats (for update after)
    home_score = match['score.fullTime.home'] or 0
    away_score = match['score.fullTime.away'] or 0
    home_xg = match['home_xg'] if not pd.isna(match['home_xg']) else 0
    away_xg = match['away_xg'] if not pd.isna(match['away_xg']) else 0
    home_poss = match['home_possession'] if not pd.isna(match['home_possession']) else 50
    away_poss = match['away_possession'] if not pd.isna(match['away_possession']) else 50

    if home_score > away_score:
        home_result = 'win'
        away_result = 'loss'
    elif home_score < away_score:
        home_result = 'loss'
        away_result = 'win'
    else:
        home_result = 'draw'
        away_result = 'draw'

    # Build feature row
    features_list.append({
        'match_id': idx,   # using loop index as ID
        'utcDate': current_date,
        'home_team': home,
        'away_team': away,
        'home_elo': home_elo,
        'away_elo': away_elo,
        'elo_diff': elo_diff,
        'home_points_last5': home_points5,
        'away_points_last5': away_points5,
        'home_xg_last5': home_xg5,
        'away_xg_last5': away_xg5,
        'home_xga_last5': home_xga5,
        'away_xga_last5': away_xga5,
        'home_possession_last5': home_poss5,
        'away_possession_last5': away_poss5,
        'home_rest_days': home_rest,
        'away_rest_days': away_rest,
        'home_matches_7d': home_m7d,
        'away_matches_7d': away_m7d,
        'home_form_ewma': home_ewma,
        'away_form_ewma': away_ewma,
        'h2h_home_win_rate': h2h_home_win_rate,
        'h2h_away_win_rate': h2h_away_win_rate,
        'h2h_avg_goals': h2h_avg_goals,
        'home_score': home_score,
        'away_score': away_score
    })

    # ---- UPDATE histories AFTER extracting features ----
    update_team_history(home, current_date, home_score, away_score,
                        home_xg, away_xg, home_poss, home_result)
    update_team_history(away, current_date, away_score, home_score,
                        away_xg, home_xg, away_poss, away_result)
    update_h2h(home, away, current_date, home_score, away_score)

# Save
features_df = pd.DataFrame(features_list)
features_df.to_csv(os.path.join(data_dir, "match_features.csv"), index=False)
print("Features saved.")