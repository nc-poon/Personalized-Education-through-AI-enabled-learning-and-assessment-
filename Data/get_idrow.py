import numpy as np
import pandas as pd

# get the start and end row of each user in the dataset

df = pd.read_pickle("")
y = df[["answered_correctly"]]
x = df.drop(["answered_correctly"], 1)
del df
last_row = x.shape[0] - 1
print(x["user_id"].nunique())
x = x.drop_duplicates(subset=["user_id"])
start = np.array(x.index.values.tolist())
end = np.append(start[1:], last_row)

pair = np.vstack((start, end)).T

np.save("", pair)