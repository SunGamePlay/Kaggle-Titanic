import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("../data/test.csv")
df = pd.concat([df_train, df_test])

# Pclass
df["Pclass"] = df["Pclass"].map(lambda x: 3 - x) #After mapping, higher values = higher chance to survive?

# Name
df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.")
df["Title"] = df["Title"].replace(["Mlle", "Ms", "Mme"], "Miss")
df["Title"] = df["Title"].replace(["Lady"], "Mrs")
df["Title"] = df["Title"].map({"Mr": -1, "Master": 1, "Miss": 2, "Mrs" : 3}).fillna(0).astype(int)
df["Title"] = df["Title"].map(lambda x: x + 1)
df = df.drop(columns = ["Name"])

# Sex
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Age
df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
df["Age"] = df["Age"].map(lambda x: 1 if x < 16 else 0)

# SibSp & Parch
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["FamilySize"] = df["FamilySize"].map({2: 1, 3: 1, 4: 1}).fillna(0).astype(int)
df = df.drop(columns = ["SibSp", "Parch"])

# Ticket
df["ConnectedSurvival"] = 0.5
for _, grp in df.groupby("Ticket"):
    if(len(grp) > 1):
        for ind, row in grp.iterrows():
            smax = grp.drop(ind)["Survived"].max()
            smin = grp.drop(ind)["Survived"].min()
            passID = row["PassengerId"]
            if (smax == 1.0):
                df.loc[df["PassengerId"] == passID, "ConnectedSurvival"] = 1
            elif (smin == 0.0):
                df.loc[df["PassengerId"] == passID, "ConnectedSurvival"] = 0
df = df.drop(columns = ["Ticket"])

# Fare
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Fare"] = df["Fare"].map(lambda x: np.log10(x) if x > 0 else 0)
df["FareGroup"] = pd.qcut(df["Fare"], 5, range(5))
df = df.drop(columns = ["Fare"])

# Cabin
df = df.drop(columns = ["Cabin"])

# Embarked
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Embarked"] = df["Embarked"].map({"S": 0, "Q": 1, "C":2}).fillna(0).astype(int)

# Features
features = df[["Pclass", "Sex", "ConnectedSurvival", "FareGroup", "Age"]]
df = pd.concat([df[["PassengerId", "Survived"]], features], axis = 1)

df_train = df[:len(df_train)]
df_test = df[len(df_train):]

x_train = df_train.drop(columns = ["PassengerId", "Survived"])
y_train = df_train["Survived"]
x_test = df_test.drop(columns = ["PassengerId", "Survived"])

x_train.to_pickle("../metadata/x_train.pkl")
y_train.to_pickle("../metadata/y_train.pkl")
x_test.to_pickle("../metadata/x_test.pkl")