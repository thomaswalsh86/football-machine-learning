import requests
import pandas as pd
import os
import pickle
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create data directory if not exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Football-Data API
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    print("Please set API_KEY in .env file")
    exit()
headers = {'X-Auth-Token': API_KEY}

base_url = "https://api.football-data.org/v4"

# Premier League competition ID
comp_id = 2021

# Seasons: 2020/21, 2021/22, 2022/23, 2023/24
seasons = [
    ("2020-08-01", "2021-05-31"),  # 2020/21
    ("2021-08-01", "2022-05-31"),  # 2021/22
    ("2022-08-01", "2023-05-31"),  # 2022/23
    ("2023-08-01", "2024-05-31"),  # 2023/24
]

# Function to save data
def save_data(df, filename):
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(df, f)
    print(f"Saved {filename}")

# Fetch matches for each season
all_matches = []
for start_date, end_date in seasons:
    url = f"{base_url}/competitions/{comp_id}/matches?dateFrom={start_date}&dateTo={end_date}"
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        matches = data['matches']
        df = pd.json_normalize(matches)
        df['season'] = f"{start_date[:4]}-{end_date[:4]}"
        all_matches.append(df)
        print(f"Fetched matches for {start_date[:4]}-{end_date[:4]}: {len(matches)}")
        time.sleep(1)  # Rate limit
    except Exception as e:
        print(f"Error fetching matches for {start_date[:4]}-{end_date[:4]}: {e}")

if all_matches:
    match_results = pd.concat(all_matches, ignore_index=True)
    save_data(match_results, "match_results.pkl")

# Fetch teams (current, but can be extended)
try:
    url = f"{base_url}/competitions/{comp_id}/teams"
    response = requests.get(url, headers=headers)
    data = response.json()
    teams = pd.json_normalize(data['teams'])
    save_data(teams, "teams.pkl")
    print(f"Fetched teams: {len(data['teams'])}")
except Exception as e:
    print(f"Error fetching teams: {e}")

# Fetch Elo ratings from ClubElo
try:
    elo_url = "https://clubelo.com/download/ENG.csv"
    response = requests.get(elo_url)
    with open(os.path.join(data_dir, "elo_ratings.csv"), 'wb') as f:
        f.write(response.content)
    print("Downloaded Elo ratings CSV")
except Exception as e:
    print(f"Error downloading Elo: {e}")