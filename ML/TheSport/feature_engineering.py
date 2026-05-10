import pandas as pd
import pickle
import os

data_dir = "data"

# Load match results
with open(os.path.join(data_dir, "match_results.pkl"), 'rb') as f:
    matches = pickle.load(f)
matches['utcDate'] = pd.to_datetime(matches['utcDate'])
matches = matches.sort_values('utcDate').reset_index(drop=True)

# Load historical ClubElo data
elo_hist = pd.read_csv(os.path.join(data_dir, "clubelo_historical.csv"))
elo_hist['date'] = pd.to_datetime(elo_hist['date'])

# Name mapping (same as in main.py)
name_mapping = {
    "Manchester United FC": "Manchester United",
    "Manchester City FC": "Manchester City",
    "Newcastle United FC": "Newcastle United",
    "Leeds United FC": "Leeds United",
    "Leicester City FC": "Leicester City",
    "Liverpool FC": "Liverpool",
    "Chelsea FC": "Chelsea",
    "Arsenal FC": "Arsenal",
    "Tottenham Hotspur FC": "Tottenham",
    "Everton FC": "Everton",
    "West Ham United FC": "West Ham United",
    "Aston Villa FC": "Aston Villa",
    "Brighton & Hove Albion FC": "Brighton",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
    "Crystal Palace FC": "Crystal Palace",
    "Southampton FC": "Southampton",
    "Brentford FC": "Brentford",
    "Nottingham Forest FC": "Nottingham Forest",
    "Fulham FC": "Fulham",
    "AFC Bournemouth": "Bournemouth",
    "Burnley FC": "Burnley",
    "Sheffield United FC": "Sheffield United",
    "Luton Town FC": "Luton Town",
    "West Bromwich Albion FC": "West Bromwich Albion",
    "Watford FC": "Watford",
    "Norwich City FC": "Norwich City",
}

# Function to get Elo of a team on a specific date (before the match)
def get_elo_before_match(team_fdo, match_date, elo_df):
    """
    Returns the most recent Elo rating for a team prior to the match date.
    
    Parameters:
        team_fdo (str): Team name as it appears in football-data.org.
        match_date (pd.Timestamp): Date of the match.
        elo_df (pd.DataFrame): Historical ClubElo data with columns 'club', 'date', 'elo'.
    
    Returns:
        float: Elo rating.
    """
    # Map football-data.org name to ClubElo name
    clubelo_name = name_mapping.get(team_fdo, team_fdo)
    
    # Filter Elo history for this team
    team_elo = elo_df[elo_df['club'] == clubelo_name].copy()
    # Keep only ratings dated before the match
    team_elo = team_elo[team_elo['date'] < match_date]
    
    if not team_elo.empty:
        # Most recent rating
        return team_elo.sort_values('date', ascending=False).iloc[0]['elo']
    else:
        # No prior Elo – find earliest rating for the team overall
        team_elo_all = elo_df[elo_df['club'] == clubelo_name]
        if not team_elo_all.empty:
            return team_elo_all.sort_values('date').iloc[0]['elo']
        # Absolute fallback
        return 1500

# Add Elo columns
matches['home_elo'] = matches.apply(
    lambda row: get_elo_before_match(row['homeTeam.name'], row['utcDate'], elo_hist), axis=1
)
matches['away_elo'] = matches.apply(
    lambda row: get_elo_before_match(row['awayTeam.name'], row['utcDate'], elo_hist), axis=1
)
matches['elo_diff'] = matches['home_elo'] - matches['away_elo']

# ============================================================
# Cumulative team stats (unchanged from original)
# ============================================================
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

for idx, match in matches.iterrows():
    home = match['homeTeam.name']
    away = match['awayTeam.name']
    home_gf = match['score.fullTime.home'] or 0
    away_gf = match['score.fullTime.away'] or 0

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