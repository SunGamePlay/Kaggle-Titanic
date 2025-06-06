# Using random forest, we had achieved an accuracy of 0.81100, which is ranked 471 / 15934 (top 3%) on the Kaggle leaderboard.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

x_train = pd.read_pickle("../metadata/x_train.pkl")
y_train = np.ravel(pd.read_pickle("../metadata/y_train.pkl"))
x_test = pd.read_pickle("../metadata/x_test.pkl")

model = RandomForestClassifier(n_estimators = 32, min_samples_split = 8)
model.fit(x_train, y_train)
scores = cross_val_score(model, x_train, y_train )

y_test = model.predict(x_test).astype(int)
submission = pd.DataFrame({
    "PassengerId": range(len(x_train) + 1, len(x_train) + len(x_test) + 1, 1),
    "Survived": y_test
})
submission.to_csv("../outputs/submission.csv", index = False)