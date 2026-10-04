import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("data/train.csv")


features = data[["Pclass", "Sex", "Age", "Fare"]].copy()

features["Age"] = features["Age"].fillna(features["Age"].median())

features["Sex"] = features["Sex"].map({
    "male": 0,
    "female": 1
})

X = features
y = data["Survived"]

model = LogisticRegression()
model.fit(X, y)


st.title("🚢 Titanic AI Predictor")

st.write("Прогноз выживания пассажира Титаника")

name = st.text_input("Имя пассажира")

age = st.slider(
    "Возраст",
    min_value=1,
    max_value=80,
    value=20
)

sex = st.selectbox(
    "Пол",
    ["Мужчина", "Женщина"]
)

pclass = st.selectbox(
    "Класс билета",
    [1, 2, 3]
)

fare = st.number_input(
    "Стоимость билета",
    min_value=0.0,
    value=50.0
)


if st.button("Предсказать"):

    if sex == "Мужчина":
        sex_number = 0
    else:
        sex_number = 1

    passenger = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex_number],
        "Age": [age],
        "Fare": [fare]
    })

    prediction = model.predict(passenger)

    if prediction[0] == 1:
        st.success(f"{name}: пассажир, скорее всего, выжил.")
    else:
        st.error(f"{name}: пассажир, скорее всего, не выжил.")