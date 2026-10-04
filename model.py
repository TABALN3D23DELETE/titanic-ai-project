import pandas as pd

data = pd.read_csv("data/train.csv")

print(data.head())

print("\nСтолбцы:")
print(data.columns)

print("\nПропущенные значения:")
print(data.isnull().sum())

features = data[["Pclass", "Sex", "Age", "Fare"]].copy()

features["Age"] = features["Age"].fillna(features["Age"].median())

features["Sex"] = features["Sex"].map({
    "male": 0,
    "female": 1
})

print("\nДанные для обучения:")
print(features.head())

print("\nПропуски после обработки:")
print(features.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = features
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nТочность модели:")
print(accuracy)

new_passenger = pd.DataFrame({
    "Pclass": [1],
    "Sex": [0],
    "Age": [20],
    "Fare": [50]
})

result = model.predict(new_passenger)

if result[0] == 1:
    print("\nПрогноз: пассажир выжил")
else:
    print("\nПрогноз: пассажир не выжил")
    

