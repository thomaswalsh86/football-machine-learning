import pandas as pd
import os

data_dir = "data"

def print_file_info(filename, description):
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.pkl'):
            df = pd.read_pickle(filepath)
        else:
            print(f"Unknown file format: {filename}")
            return
        print(f"\n{'='*60}")
        print(f"{description} ({filename})")
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("First 3 rows:")
        print(df.head(3))
    else:
        print(f"\n{description} ({filename}) not found – run the corresponding script first.")

# 1. Raw match results (from FBref, many seasons)
print_file_info("match_results.pkl", "Match Results (with xG, possession)")

# 2. Custom Elo history (from calculate_elo.py)
print_file_info("elo_history.csv", "Elo History (pre‑match Elo for each fixture)")

# 3. Current Elo ratings snapshot
print_file_info("current_elo.csv", "Current Elo Ratings")

# 4. Final feature table (used by train_model.py)
print_file_info("match_features.csv", "Engineered Match Features (for ML)")

# 5. Teams list – only if it exists (optional)
teams_path = os.path.join(data_dir, "teams.pkl")
if os.path.exists(teams_path):
    teams = pd.read_pickle(teams_path)
    print(f"\n{'='*60}")
    print("Teams (teams.pkl)")
    print(f"Shape: {teams.shape}")
    print("Columns:", teams.columns.tolist())
    print(teams.head(3))