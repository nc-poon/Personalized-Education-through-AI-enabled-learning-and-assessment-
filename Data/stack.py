import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler

train = pd.read_csv("", index_col=0)
print(train.info())
cols = [
    "user_id",
    "answered_correctly",
    "prior_question_elapsed_time",
    "prior_question_had_explanation",
    "lecture_tag_mean",
    "question_tag_mean",
    "user_pct_correct",
    "content_mean",
    "content_std",
    "content_count",
]
dtypes = {
    "user_id": "int32",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32",
    "prior_question_had_explanation": "int8",
    "lecture_tag_mean": "float32",
    "question_tag_mean": "float32",
    "user_pct_correct": "float32",
    "content_mean": "float32",
    "content_std": "float32",
    "content_count": "float32",
}

for file in os.listdir("\\"):
    if file.endswith(".csv") & file.startswith("Dataset"):
        print(file)

        add = pd.read_csv(file, index_col=0)
        print("stacking.......")
        train = np.vstack([train, add])
        print("done")
        print(train.shape)


train = pd.DataFrame(train, columns=cols)
train = train.astype(dtypes)

scaler = RobustScaler()
train[
    ["content_count", "lecture_tag_mean", "question_tag_mean"]
] = scaler.fit_transform(
    train[["content_count", "lecture_tag_mean", "question_tag_mean"]]
).astype(
    "float32"
)

print(train.info())
print(train.describe())
print(train.head())
print(train.tail())

pd.to_pickle(train, "")