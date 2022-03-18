from xgboost import XGBClassifier  # xgboost
import xgboost as xgb
import pandas as pd
import matplotlib as plt

train = pd.read_pickle("")

y_train = train["answered_correctly"]
x_train = train.drop(columns=["answered_correctly", "user_id"])
del train

model = XGBClassifier(
    learning_rate=0.3, n_estimators=150, max_depth=7, min_child_weight=5
)

model.fit(x_train, y_train)


model.save_model("")

xgb.plot_importance(model)
plt.rcParams["figure.figsize"] = [5, 5]
plt.pyplot.show()