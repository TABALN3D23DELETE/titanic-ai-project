# 🚢 Titanic AI Predictor

Проект машинного обучения для предсказания выживания пассажиров Титаника.

Датасет взят с платформы Kaggle из соревнования **Titanic - Machine Learning from Disaster**.

## Возможности

Пользователь вводит данные пассажира:

- имя;
- возраст;
- пол;
- класс билета;
- стоимость билета.

После нажатия кнопки **«Предсказать»** модель определяет, выжил бы пассажир или нет.

## Машинное обучение

Для обучения используется модель **Logistic Regression**.

В качестве признаков используются:

- `Pclass` — класс пассажира;
- `Sex` — пол;
- `Age` — возраст;
- `Fare` — стоимость билета.

Точность модели на тестовой выборке составляет примерно **80%**.

## Технологии

- Python
- Pandas
- Scikit-learn
- Streamlit
- Kaggle
- GitHub

## Структура проекта

```text
titanic-ai-project/
├── data/
│   ├── train.csv
│   └── test.csv
├── app.py
├── model.py
├── requirements.txt
└── README.md
```

## Запуск проекта

Установить необходимые библиотеки:

```bash
pip install -r requirements.txt
```

Запустить приложение:

```bash
python -m streamlit run app.py
```

После запуска приложение откроется в браузере.

## Dataset

Titanic - Machine Learning from Disaster, Kaggle.