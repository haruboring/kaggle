import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("./input/train.csv")
print(train_data.head())
test_data = pd.read_csv("./input/test.csv")
print(test_data.head())

# カテゴリーのdummiesと、ツイート長より推測。なんで0になるんだ、せめて少数であってくれ。
# randomForestのパラメータ初期値にするだけで0以外も出た。大事なんやなぁ。→ゴミみたいな結果
# こ

y = train_data["likes"]
y = y.astype("float")
print(y.head())

features = ["keyword"]
categories = train_data["keyword"].unique()
print(categories[:10])

X = pd.Categorical(train_data["keyword"], categories=categories)
X_test = pd.Categorical(test_data["keyword"], categories=categories)
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
print("")
print(X.head())
print("")
print(X_test.head())
X["tweet_lengths"] = train_data["tweets"].str.len()
X_test["tweet_lengths"] = test_data["tweets"].str.len()

model = RandomForestClassifier()
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({"id": test_data.id, "likes": predictions})
output.to_csv("output/keyword_dummies_tweet_length.csv", index=False)
print("Your submission was successfully saved!")
