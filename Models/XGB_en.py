from xgboost import XGBClassifier  # xgboost
import xgboost as xgb
import numpy as np
import matplotlib as plt


pred = np.load("")  # output from different models
actual = np.load("")

print("finished loading")

model = XGBClassifier(
    learning_rate=0.3, n_estimators=50, max_depth=4, min_child_weight=3
)

model.fit(pred, actual)

model.save_model("")

xgb.plot_importance(model)
plt.rcParams["figure.figsize"] = [5, 5]
plt.pyplot.show()