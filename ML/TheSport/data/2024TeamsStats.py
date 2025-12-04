import soccerdata as sd
import pandas as pd
import matplotlib.pyplot as plt


#build csv

#write data, teams as rows & stats as columns
#shots
#goals

#teams =[]
fbref = sd.FBref(leagues=["ENG-Premier League"],seasons=["2023","2024","2025"])
shooting = fbref.read_team_season_stats(stat_type="shooting")
print(shooting.columns)
print(shooting.head)

stat = shooting.groupby("team").agg(
    goals_pg=(('Standard', 'Gls'), 'mean'),
    shots_pg=(('Standard', 'Sh'), 'mean')
)



plt.scatter(stat["shots_pg"], stat["goals_pg"])
plt.title("Shots vs Goals (Premier League) 2025")
plt.xlabel("Shots per Match")
plt.ylabel("Goals per Match")


for team, row in stat.iterrows():
    plt.annotate(team, (row["shots_pg"], row["goals_pg"]))

plt.show()
