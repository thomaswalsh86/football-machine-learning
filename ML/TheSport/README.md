# Sports Match Prediction Data Sources

This README outlines the various data sources required for building a comprehensive football (soccer) match prediction model, based on factors like Elo ratings, team statistics, match history, and contextual elements. The model aims to predict match outcomes (win/draw/loss) and goals, drawing from machine learning techniques (e.g., XGBoost) and domain knowledge from betting rules and academic research.

Data is primarily sourced from free, publicly available platforms. We'll use Python libraries like `soccerdata` for automated fetching, with manual scraping or APIs for additional data.

## Core Data Sources

### 1. **Team and Match Statistics (FBref via soccerdata)**
   - **Description**: Comprehensive match-level and season-level stats including shooting (goals, shots, XG), defense (goals conceded, XG conceded), possession, passing, and more. Essential for calculating team performance metrics, recent form, and head-to-head records.
   - **Where to Find**: 
     - Website: [FBref.com](https://fbref.com/) (StatsBomb data).
     - Access: Via `soccerdata` Python library (`pip install soccerdata`). Supports multiple leagues (e.g., Premier League) and seasons.
     - Example Usage: `fbref = sd.FBref(leagues=["ENG-Premier League"], seasons=["2024"])` then `fbref.read_team_season_stats(stat_type="shooting")`.
   - **Key Features Extractable**: Goals scored/conceded, XG, possession %, shots, win/loss records, home/away splits.

### 2. **Elo Ratings**
   - **Description**: Elo-based team strength ratings, updated after each match. Critical for determining team quality differences, which is a top contributor to match outcomes per research.
   - **Where to Find**:
     - [ClubElo.com](https://clubelo.com/) or [eloratings.net](https://www.eloratings.net/): Historical Elo data for download (CSV/JSON).
     - Python Library: `football-elo` (`pip install football-elo`) for calculation and updates.
     - Alternative: Implement custom Elo system using match results (K-factor=30 for soccer).
   - **Key Features Extractable**: Current Elo, Elo difference between teams, historical Elo trends.

### 3. **Match Schedules and Results**
   - **Description**: Dates, scores, and fixtures for calculating rest days, recency of games, and match importance.
   - **Where to Find**:
     - [FBref.com](https://fbref.com/): Match logs and schedules.
     - Access: Via `soccerdata` (`fbref.read_schedule()` or `fbref.read_match_results()`).
     - Alternative: [Transfermarkt.com](https://www.transfermarkt.com/) or league official sites (e.g., [Premier League](https://www.premierleague.com/)).
   - **Key Features Extractable**: Match dates, rest days between games, next game importance.

### 4. **Stadium Locations and Travel Distance**
   - **Description**: Geographic data for calculating travel distance, which affects team performance (fatigue, home advantage).
   - **Where to Find**:
     - [Wikipedia](https://en.wikipedia.org/wiki/List_of_stadiums_in_England) or team-specific pages for stadium coordinates.
     - APIs: Google Maps Geocoding API (requires API key) or OpenStreetMap via `geopy` library (`pip install geopy`).
     - Pre-built Datasets: Search for "football stadium coordinates CSV" on GitHub or Kaggle.
   - **Key Features Extractable**: Latitude/longitude of home stadiums, distance calculations (use `geopy.distance`).

### 5. **Manager Data (Tenure and History)**
   - **Description**: Number of matches since manager appointment, influencing team strategy and performance.
   - **Where to Find**:
     - [Transfermarkt.com](https://www.transfermarkt.com/): Manager history per team.
     - [Wikipedia](https://en.wikipedia.org/): Team pages with manager sections.
     - Access: Manual scraping with `BeautifulSoup` (`pip install beautifulsoup4`) or Selenium for dynamic sites.
   - **Key Features Extractable**: Manager start date, matches under current manager.

### 6. **Player Data (Rotations and Lineups)**
   - **Description**: Starting lineups and player rotations to assess lineup stability and fatigue.
   - **Where to Find**:
     - [FBref.com](https://fbref.com/): Player match stats and lineups.
     - Access: Via `soccerdata` (`fbref.read_player_match_stats()`).
     - Alternative: [Transfermarkt.com](https://www.transfermarkt.com/) for squad data.
   - **Key Features Extractable**: % of lineup changes from previous match, player availability.

### 7. **League Standings and Contextual Factors**
   - **Description**: Current league positions, points, and other metadata (e.g., match importance).
   - **Where to Find**:
     - [FBref.com](https://fbref.com/): League tables.
     - Access: Via `soccerdata` (`fbref.read_league_table()`).
     - Official Sites: League websites (e.g., Premier League for standings).

## Additional Tools and Libraries
- **Data Fetching**: `soccerdata` (primary), `requests` for APIs, `BeautifulSoup`/`Selenium` for scraping.
- **Geospatial**: `geopy` for distances.
- **Caching**: `joblib` or `pickle` to store fetched data locally.
- **ML/Modeling**: `xgboost`, `scikit-learn`, `shap` for feature analysis.
- **Data Processing**: `pandas`, `numpy`.

## Data Collection Workflow
1. **Setup**: Install libraries (`pip install soccerdata geopy beautifulsoup4 requests xgboost shap`).
2. **Fetch Core Stats**: Use `soccerdata` for FBref data.
3. **Supplement**: Scrape or API-call for Elo, stadiums, managers.
4. **Cache**: Save DataFrames to avoid re-fetching.
5. **Aggregate**: Build per-match feature sets (e.g., Elo diff, travel distance).
6. **Train/Test**: Use historical seasons for model training.

## Notes
- **Legal/Ethical**: Ensure compliance with terms of service; avoid overloading free APIs.
- **Data Quality**: Cross-verify sources; handle missing data (e.g., impute Elo for new teams).
- **Updates**: Elo and stats change post-match—implement daily/weekly refreshes.
- **Alternatives**: For paid data, consider StatsBomb API or Opta.

For implementation code, see `main.py` and `data/2024TeamsStats.py`.