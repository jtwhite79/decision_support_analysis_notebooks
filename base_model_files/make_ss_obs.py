import pandas as pd

df = pd.read_csv("obs_data.csv",parse_dates=["datetime"])
fo_mean = df.loc[df.site=="fo_gage_1","value"].mean()
print(fo_mean)
df = df.loc[df.site.apply(lambda x: x.startswith("tr"))]
means = df.groupby("site").mean()
means.loc[:,"r"] = means.index.map(lambda x: int(x.split('_')[1])+1)
means.loc[:,"c"] = means.index.map(lambda x: int(x.split('_')[2])+1)
means.loc[:,"l"] = 3
means.to_csv("ss_gw_obs.csv")
print(means)