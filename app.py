import streamlit as st

st.title("Моё первое приложение 🚀")
st.write("Приложение работает!")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# загрузка
df = pd.read_csv("data.csv")

# удаляем ненужные колонки
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

# преобразуем категории в числа
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# удаляем пропуски
df = df.dropna()

# целевая переменная
target_column = "Survived"

# делим данные
X = df.drop(target_column, axis=1)
y = df[target_column]

# train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# модель
model = RandomForestClassifier()
model.fit(X_train, y_train)

# сохраняем
joblib.dump(model, "model.joblib")

print("Модель обучена и сохранена!")
print("Используемые признаки:", X.columns)

import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.joblib")

st.title("Titanic Survival Prediction")

st.write("Введите данные пассажира:")

PassengerId = st.number_input("PassengerId")
Pclass = st.number_input("Pclass (1-3)")
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.number_input("Age")
SibSp = st.number_input("SibSp")
Parch = st.number_input("Parch")
Fare = st.number_input("Fare")
Embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# преобразование
Sex = 0 if Sex == "male" else 1
Embarked = {"S": 0, "C": 1, "Q": 2}[Embarked]

if st.button("Predict"):
    data = pd.DataFrame([[
        PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    ]],
    columns=["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"])

    result = model.predict(data)

    if result[0] == 1:
        st.success("Выживет ✅")
    else:
        st.error("Не выживет ❌")
