# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from sklearn.ensemble import RandomForestClassifier

for dirname, _, filenames in os.walk("./input/titanic"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("./input/titanic/train.csv")
print(train_data.head())  # 表示する
test_data = pd.read_csv("./input/titanic/test.csv")
print(test_data.head())  # 表示する

women = train_data.loc[train_data.Sex == "female"]["Survived"]
rate_women = sum(women) / len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == "male"]["Survived"]
rate_men = sum(men) / len(men)

print("% of men who survived:", rate_men)


# 実際にモデル作る

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
print(train_data[features].head())
print("")
print(X.head())  # 表示する
X_test = pd.get_dummies(test_data[features])
print(test_data[features].head())
print("")
print(X_test.head())  # 表示する

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
output.to_csv("output/submission.csv", index=False)
print("Your submission was successfully saved!")
