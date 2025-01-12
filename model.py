from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

script_dir = Path(__file__).parent  # لمكتبة pathlib
Path = script_dir / "assets" / "diabetes.csv"
df = pd.read_csv(Path)

X = df[
    ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', "BMI", "DiabetesPedigreeFunction", "Age"]]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# y_pred = model.predict(X_test)

def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    data = {
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        "BMI": [BMI],
        "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
        "Age": [Age]
    }
    dataframe_test = pd.DataFrame(data)
    result = model.predict(dataframe_test)
    result = "positive" if result[0] == 1 else "negative"
    return result


if __name__ == '__main__':
    result = predict(10, 20, 30, 50, 60, 70, 80, 40)
    print(result)
