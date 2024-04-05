from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np

df = pd.read_parquet("./data/prepared_dataset.parquet",engine="pyarrow")
y = df["phase"]
X = df.drop("phase", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.1,
    'max_depth': 24
}

model = xgb.train(params, dtrain)

y_pred = model.predict(dtest)
print(pd.Series(y_pred))
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Médio Quadrático (MSE): {mse}')