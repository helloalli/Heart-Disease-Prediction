from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

MODEL_FILE = "heart_model.pkl"
COLUMNS_FILE = "model_columns.pkl"

# ----------------------------
# TRAIN MODEL IF NOT EXISTS
# ----------------------------
if not os.path.exists(MODEL_FILE):

    print("Training model...")

    df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\project2\heart.csv")
    df = df[df["RestingBP"] != 0]
    mask = df["HeartDisease"] == 0

    df.loc[mask, "Cholesterol"] = df.loc[mask, "Cholesterol"].replace(
        0, df.loc[mask, "Cholesterol"].median()
    )

    df.loc[~mask, "Cholesterol"] = df.loc[~mask, "Cholesterol"].replace(
        0, df.loc[~mask, "Cholesterol"].median()
    )
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=417
    )

    model = RandomForestClassifier(n_estimators=200, random_state=417)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(X.columns, COLUMNS_FILE)

    print("Model saved successfully!")

else:
    print("Loading existing model...")
model = joblib.load(MODEL_FILE)
model_columns = joblib.load(COLUMNS_FILE)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "Age": int(request.form["Age"]),
            "RestingBP": int(request.form["RestingBP"]),
            "Cholesterol": int(request.form["Cholesterol"]),
            "FastingBS": int(request.form["FastingBS"]),
            "MaxHR": int(request.form["MaxHR"]),
            "Oldpeak": float(request.form["Oldpeak"]),
            "Sex_M": 1 if request.form["Sex"] == "M" else 0,
        }

        input_df = pd.DataFrame([data])

        # Ensure same feature order
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            result = f" High Risk ({probability*100:.2f}%)"
        else:
            result = f" Low Risk ({(1-probability)*100:.2f}%)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)