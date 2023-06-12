import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("./input/train.csv")
print(train_data.head())
test_data = pd.read_csv("./input/test.csv")
print(test_data.head())

# Tweetの長さのみから予測するモデル→

y = train_data["likes"]
print(y.head())

X = pd.DataFrame()
X["tweet_lengths"] = train_data["tweets"].str.len()

print("")
print(X.head())
X_test = pd.DataFrame()
X_test["tweet_lengths"] = test_data["tweets"].str.len()
print("")
print(X_test.head())

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({"is": test_data.id, "likes": predictions})
output.to_csv("output/tweet_len.csv", index=False)
print("Your submission was successfully saved!")
