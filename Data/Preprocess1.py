import pandas as pd
from sklearn.preprocessing import MinMaxScaler


dtypes = {
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32",
    "prior_question_had_explanation": "int8",
}

cols = [
    "timestamp",
    "user_id",
    "content_id",
    "answered_correctly",
    "prior_question_elapsed_time",
    "prior_question_had_explanation",
]

print("reading......")

train = pd.read_csv("train.csv", usecols=cols, dtype=dtypes)

print("done reading, filling na....")

### FILL NA ###
train["prior_question_elapsed_time"] = train["prior_question_elapsed_time"].fillna(0)
train["prior_question_had_explanation"] = (
    train["prior_question_had_explanation"].fillna(0).astype("int8")
)

print("na filled, removing outliers....")

# remove outliers / reduce data size #

cols = ["timestamp"]

Q1 = train[cols].quantile(0.25)
Q3 = train[cols].quantile(0.75)
IQR = Q3 - Q1


train = train[~((train[cols] > (Q3 + 1.5 * IQR))).any(axis=1)].reset_index(drop=True)
train = train.drop("timestamp", 1)

print("removed outliers, normalising....")

### NORMALISE ###

scaler = MinMaxScaler()
train["prior_question_elapsed_time"] = scaler.fit_transform(
    train[["prior_question_elapsed_time"]]
).astype("float32")

print("normalised, preprocessing questions.....")

### GET QUESTION FEATURES ###

# Group by content
qf = (
    train[train["answered_correctly"] != -1]
    .groupby("content_id")
    .agg({"answered_correctly": ["mean", "std", "count"]})
    .reset_index()
)

qf.columns = ["content_id", "content_mean", "content_std", "content_count"]

print("questions data info")
print(qf.head())
print(qf.info())

pd.to_pickle(qf, "qfeatures.pkl")


del qf

print("done, preprocessing lectures......")

### GET LECTURE FEATURES ###

# Group by content
lf = (
    train[train["answered_correctly"] == -1]
    .groupby("content_id")
    .size()
    .reset_index(name="lecture_count")
)

lf.columns = ["content_id", "lecture_count"]
print("lecture data and train info")
print(lf.head())
print(lf.describe())
print(train.head())
print(train.describe())
pd.to_pickle(lf, "lfeatures.pkl")
train.to_csv("PPtrain.csv")

del lf
del train

print("done, getting unique tags.....")
### GET TAGS ###

questions = pd.read_csv("questions.csv")

questions["tags"] = questions["tags"].dropna()
questions = questions.drop(["bundle_id", "correct_answer", "part"], 1)
questions["tags"] = questions["tags"].astype(str)

tags = [x.split() for x in questions[questions.tags != "nan"].tags.values]
tags = [item for elem in tags for item in elem]
tags = set(tags)
tags = list(tags)
df = pd.DataFrame(tags)
df.to_csv("tags.csv")
print(f"There are {len(tags)} different tags")


### END ###