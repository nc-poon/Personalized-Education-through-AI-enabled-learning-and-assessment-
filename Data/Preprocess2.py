import pandas as pd
import numpy as np


dtypes = {
    "user_id": "int32",
    "content_id": "int16",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32",
    "prior_question_had_explanation": "int8",
}

cols = [
    "user_id",
    "content_id",
    "answered_correctly",
    "prior_question_elapsed_time",
    "prior_question_had_explanation",
]

qf = pd.read_csv("qfeatures.csv")
lf = pd.read_csv("lfeatures.csv")
tags_df = pd.read_csv("tags.csv")
tag_list = tags_df["0"].values.tolist()
questions = pd.read_csv("questions.csv")
questions["tags"] = questions["tags"].dropna()
questions = questions.drop(["bundle_id", "correct_answer", "part"], 1)


start = 0
nrow = 1 * 10 ** 4  # Batch Size
lrow = 91732845  # Size of dataset

### start preprossing in batches ###
while (start + nrow) < lrow:
    print("reading data......")
    if start == 0:
        train = pd.read_csv(
            "PPtrain.csv", usecols=cols, dtype=dtypes, skiprows=start, nrows=nrow
        )

    else:
        train = pd.read_csv(
            "PPtrain.csv",
            usecols=cols,
            dtype=dtypes,
            skiprows=range(1, start),
            nrows=nrow,
        )

    last = nrow - 1
    last_u = train["user_id"][last]

    while train["user_id"][last] == last_u:
        last = last - 1

    train = train.iloc[: last + 1, :]
    start = start + last + 2

    ### Add columns for each tags with value 0 ###
    tags = pd.DataFrame(0, index=np.arange(len(train)), columns=tag_list, dtype="int8")
    colnames = train.columns.values.tolist() + tags.columns.values.tolist()
    print("Done reading , stacking train and tags")
    train2 = pd.DataFrame(np.hstack([train, tags]))
    train2.columns = colnames
    train2.columns = train2.columns.astype(str)
    print("done stacking")
    print("Compiling user questions history")

    def qstat(row):
        id = int(train2.iloc[row]["content_id"])
        correct = bool(train2.iloc[row]["answered_correctly"])
        try:
            qtag = questions.iloc[id]["tags"]
        except IndexError:  # some content has no tags
            return
        tags = qtag.split()

        for i in tags:
            i = str(i)
            if correct:
                train2.at[row, i] += 1  # Add 1 to relevant tags if correct
            else:
                train2.at[row, i] -= 1  # Minus 1 to relevant tags if correct

    def lstat(row):
        id = int(train2.iloc[row]["content_id"])

        try:
            qtag = questions.iloc[id]["tags"]
        except IndexError:
            return
        tags = qtag.split()

        for i in tags:

            i = str(i)
            train2.at[row, i] += 1

    def comtag(row, type):

        id = int(train2.iloc[row]["content_id"])
        try:
            qtag = questions.iloc[id]["tags"]
        except IndexError:
            return
        tags = qtag.split()
        total = []
        for t in tags:
            total.append(int(train2.iloc[row][str(t)]))
        if type == 0:
            train2.lecture_tag_mean.iloc[row] = sum(total) / len(
                total
            )  # Get mean values

        else:
            train2.question_tag_mean.iloc[row] = sum(total) / len(total)

    ### Get question tag mean ###

    train2["question_tag_mean"] = 0
    n_col = train2.shape[1] - 1

    for row in train2.index:

        if train2.iloc[row]["answered_correctly"] == -1:
            continue
        else:

            if train2.iloc[row]["user_id"] == train2.iloc[row - 1]["user_id"]:
                train2.loc[row, 5:n_col] = train2.iloc[row - 1, 5:n_col].values

            qstat(row)
            comtag(row, 1)

    train2 = train2[
        [
            "user_id",
            "content_id",
            "answered_correctly",
            "prior_question_elapsed_time",
            "prior_question_had_explanation",
            "question_tag_mean",
        ]
    ]

    print("Done with questions tag mean , stacking train and tags")
    colnames = train2.columns.values.tolist() + tags.columns.values.tolist()
    train2 = pd.DataFrame(np.hstack([train2, tags]))
    train2.columns = colnames
    train2.columns = train2.columns.astype(str)
    print("done stacking")

    print("Compiling user lectures history")

    ### Get lecture tag mean ###

    train2["lecture_tag_mean"] = 0
    n_col = train2.shape[1] - 1

    for row in train2.index:

        if train2.iloc[row]["user_id"] == train2.iloc[row - 1]["user_id"]:

            train2.loc[row, 6:n_col] = train2.iloc[row - 1, 6:n_col].values

        if train2.iloc[row]["answered_correctly"] == -1:

            lstat(row)
        comtag(row, 0)

    train2 = train2[
        [
            "user_id",
            "content_id",
            "answered_correctly",
            "prior_question_elapsed_time",
            "prior_question_had_explanation",
            "question_tag_mean",
            "lecture_tag_mean",
        ]
    ]

    train2 = train2[train2.answered_correctly != -1]

    print("calculating percentage correct")

    ### calculate percentage correct ###

    train2["sum"] = train2.groupby("user_id")["answered_correctly"].cumsum()
    train2["count"] = train2.groupby("user_id")["answered_correctly"].cumcount() + 1
    train2["user_pct_correct"] = train2["sum"] / train2["count"]

    train2 = train2.drop(columns=["sum", "count"])

    print("done")

    ### add content stats ###

    print("mapping content")

    train2["content_mean"] = train2["content_id"].map(qf["content_mean"])

    train2["content_std"] = train2["content_id"].map(qf["content_std"])

    train2["content_count"] = train2["content_id"].map(qf["content_count"])

    train2 = train2[
        [
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
    ]
    train2 = train2.fillna("0")

    # normalise if not processed in batches

    """scaler = RobustScaler()
    train2[
        ["content_count", "lecture_tag_mean", "question_tag_mean"]
    ] = scaler.fit_transform(
        train2[
            ["content_count",  "lecture_tag_mean", "question_tag_mean"]
        ]
    ).astype(
        "float32"
    )"""

    train2[["answered_correctly", "prior_question_had_explanation"]] = train2[
        ["answered_correctly", "prior_question_had_explanation"]
    ].astype("int8")
    train2[["content_mean", "content_std"]] = train2[
        ["content_mean", "content_std"]
    ].astype("float32")

    print(train2.head())
    print(train2.describe())
    print(train2.info())
    print(start)
    train2.to_csv("Dataset" + str(start) + ".csv")
