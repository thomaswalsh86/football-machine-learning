import pandas as pd
import pickle
import os
import numpy as np
import soccerdata as sd

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

seasons = [
    "1718", "1819", "1920", "2021",
    "2122", "2223", "2324", "2425"
]

def season_to_understat_year(season_code):
    return 2000 + int(season_code[:2])

# ------------------------------------------------------------
# Team name mapping: FBref → Understat (common differences)
# ------------------------------------------------------------
name_map = {
    "Manchester Utd": "Manchester United",
    "Newcastle Utd": "Newcastle United",
    "Wolves": "Wolverhampton Wanderers",
    "Brighton": "Brighton and Hove Albion",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Tottenham": "Tottenham Hotspur",
    "Leicester City": "Leicester City",
    "Leeds United": "Leeds United",
    "Norwich City": "Norwich City",
    "Sheffield Utd": "Sheffield United",
    "Cardiff City": "Cardiff City",
    "Huddersfield Town": "Huddersfield Town",
    "Fulham": "Fulham",
    "Burnley": "Burnley",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Southampton": "Southampton",
    "Watford": "Watford",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Liverpool": "Liverpool",
    "Manchester City": "Manchester City",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Nott'ham Forest": "Nottingham Forest",
}

def normalize_team(name):
    """Map FBref team name to Understat-compatible name."""
    return name_map.get(name, name)

all_matches = []
for season in seasons:
    try:
        # 1. FBref schedule
        fbref = sd.FBref(leagues="ENG-Premier League", seasons=season)
        schedule = fbref.read_schedule(force_cache=False)
        if schedule is None or schedule.empty:
            print(f"No schedule for {season}")
            continue

        # 2. Understat schedule (real xG)
        us_year = season_to_understat_year(season)
        understat = sd.Understat(leagues="ENG-Premier League", seasons=us_year)
        us_schedule = understat.read_schedule(force_cache=False)

        # Prepare Understat data for merge
        us_xg = us_schedule[['date', 'home_team', 'away_team', 'home_xg', 'away_xg']].copy()
        us_xg['date'] = pd.to_datetime(us_xg['date']).dt.date
        us_xg['home_team'] = us_xg['home_team'].apply(normalize_team)
        us_xg['away_team'] = us_xg['away_team'].apply(normalize_team)

        # Normalise FBref names
        schedule['date'] = pd.to_datetime(schedule['date']).dt.date
        schedule['home_team'] = schedule['home_team'].apply(normalize_team)
        schedule['away_team'] = schedule['away_team'].apply(normalize_team)

        # Merge on date + teams
        schedule = schedule.merge(us_xg,
                                  left_on=['date', 'home_team', 'away_team'],
                                  right_on=['date', 'home_team', 'away_team'],
                                  how='left')

        schedule['season'] = season
        all_matches.append(schedule)
        print(f"Fetched {season}: {len(schedule)} matches (with real xG)")
    except Exception as e:
        print(f"Error fetching {season}: {e}")

if all_matches:
    match_results = pd.concat(all_matches, ignore_index=True)

    # Parse combined score if needed
    if 'score' in match_results.columns:
        score_split = match_results['score'].str.split(r'[–-]', expand=True)
        match_results['home_score'] = pd.to_numeric(score_split[0], errors='coerce')
        match_results['away_score'] = pd.to_numeric(score_split[1], errors='coerce')

    # Rename columns
    rename_map = {
        'date': 'utcDate',
        'home_team': 'homeTeam.name',
        'away_team': 'awayTeam.name',
        'home_score': 'score.fullTime.home',
        'away_score': 'score.fullTime.away'
    }
    match_results = match_results.rename(columns=rename_map)

    # Ensure required columns exist (possession still placeholder)
    for col, default in [('home_xg', np.nan), ('away_xg', np.nan),
                         ('home_possession', 50.0), ('away_possession', 50.0)]:
        if col not in match_results.columns:
            match_results[col] = default

    # ------------------------------------------------------------
    # Safe forward‑fill of missing xG (per team, using past matches)
    # ------------------------------------------------------------
    match_results = match_results.sort_values('utcDate').reset_index(drop=True)

    # For home teams
    last_home_xg = {}
    for idx, row in match_results.iterrows():
        team = row['homeTeam.name']
        if pd.isna(row['home_xg']):
            match_results.at[idx, 'home_xg'] = last_home_xg.get(team, np.nan)
        else:
            last_home_xg[team] = row['home_xg']

    # For away teams
    last_away_xg = {}
    for idx, row in match_results.iterrows():
        team = row['awayTeam.name']
        if pd.isna(row['away_xg']):
            match_results.at[idx, 'away_xg'] = last_away_xg.get(team, np.nan)
        else:
            last_away_xg[team] = row['away_xg']

    # If still NaN (first match of a team), fill with league average xG (≈ 1.4)
    match_results['home_xg'] = match_results['home_xg'].fillna(1.4)
    match_results['away_xg'] = match_results['away_xg'].fillna(1.2)

    # Keep needed columns
    needed_cols = [
        'utcDate', 'homeTeam.name', 'awayTeam.name',
        'score.fullTime.home', 'score.fullTime.away',
        'home_xg', 'away_xg', 'home_possession', 'away_possession',
        'season'
    ]
    match_results = match_results[needed_cols]
    match_results['utcDate'] = pd.to_datetime(match_results['utcDate']).dt.tz_localize(None)

    # Save
    with open(os.path.join(data_dir, "match_results.pkl"), 'wb') as f:
        pickle.dump(match_results, f)

    # ───── Final print block ─────
    print("\n" + "="*60)
    print("Data collection complete!")
    print(f"Total matches saved: {len(match_results)}")
    print(f"Seasons: {match_results['season'].unique().tolist()}")
    print(f"Date range: {match_results['utcDate'].min().date()} → {match_results['utcDate'].max().date()}")
    print("\nFirst 5 matches:")
    print(match_results[['utcDate', 'homeTeam.name', 'awayTeam.name',
                         'score.fullTime.home', 'score.fullTime.away',
                         'home_xg', 'away_xg']].head())
    print("\nMissing xG after filling:")
    print(match_results[['home_xg', 'away_xg']].isna().sum())
    print("="*60)
else:
    print("No match data fetched. Exiting.")