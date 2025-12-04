import soccerdata as sd
import panda as pd
import logging

fbref = sd.FBref(leagues=["ENG-Premier League"], seasons=[2025])
stats = fbref.read_team_season_stats(stat_type="shooting")

logging.getLogger().setLevel(logging.ERROR)

print(stats)