from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
#import seaborn as sns
import pandas as pd
import numpy as np
import random
import os

path = os.path.join(os.getcwd(),"data","train_data.parquet")
#path = "train_data.parquet"
df =  pd.read_parquet(path, engine='pyarrow')
print(len(df))

df["ther_area"].replace(["A","B","C","D","G","H","J","L","M","N","P","R","S","V"],[1,2,3,4,5,6,7,8,9,10,11,12,13,14], inplace=True)
df["ther_area"] = df["ther_area"].astype(float).fillna(0.0)

df["main_channel"].replace(["RETAIL","HOSPITAL","COMBINED","OTHERS"],[1,2,3,4], inplace= True)
df["main_channel"] = df["main_channel"].astype(float).fillna(0.0)

df["brandID"] = df["brand"].replace(df["brand"].unique(), range(1,259))
df["countryID"] = df["country"].replace(df["country"].unique(),range(1,31))

df["hospital_rate"].replace(np.nan, 0.0, inplace=True)
df["year"] = [i.split("-")[0] for i in df["date"].astype(str)]
df["year"] = df["year"].astype(int)

brand = df["brand"]
del df["brand"]
country = df["country"]
del df["country"]
brandID = df["brandID"]
del df["brandID"]
countryID = df["countryID"]
del df["countryID"]
year = df["year"]
del df["year"]
del df["date"]

forest = IsolationForest(n_estimators= 45, contamination = float(0.05), random_state = 42, warm_start= True)
forest.fit(df)

prediction = forest.predict(df)

outliers = pd.Series(prediction)
soma = outliers.value_counts().sum()
outlierNum = outliers.value_counts()[-1]
print(outlierNum/soma)

pca = PCA(n_components = 2)
transform = pca.fit_transform(df)

pca.explained_variance_ratio_

X = []
Y = []
for i in transform:
  X.append(i[0])
  Y.append(i[1])

dfPlot = pd.DataFrame({"X":X,"Y":Y,"outlier":outliers})

#sns.scatterplot(data = dfPlot, x = dfPlot["X"],y = dfPlot["Y"],hue = dfPlot["outlier"])

dfParaquet = pd.DataFrame({"brand":brandID,"country":countryID,"phase":df["phase"],"dayweek":df["dayweek"],"month":df["month"],"wd_perc":df["wd_perc"],"ther_area":df["ther_area"],"hospital_rate":df["hospital_rate"],"n_nwd_bef":df["n_nwd_bef"],"n_nwd_aft":df["n_nwd_aft"],"n_weekday_0":df['n_weekday_0'],'n_weekday_1':df["n_weekday_1"],'n_weekday_2':df["n_weekday_2"],'n_weekday_3':df["n_weekday_3"],'n_weekday_4':df['n_weekday_4'],"year":year,"wd":df["wd"],"wd_left":df["wd_left"],"monthly":df["monthly"],"main_channel":df["main_channel"],"outlier":prediction})

dfParaquet["outlier"].value_counts()

dfParaquet = dfParaquet[dfParaquet['outlier'] != -1]

del dfParaquet["outlier"]

print(dfParaquet)

dfParaquet.to_parquet('prepared_dataset.parquet', index=False)