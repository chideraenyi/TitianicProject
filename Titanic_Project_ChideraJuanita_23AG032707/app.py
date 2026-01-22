from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/titanic_survival_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        pclass = int(request.form["pclass"])
        sex = int(request.form["sex"])
        age = float(request.form["age"])
        fare = float(request.form["fare"])
        embarked = int(request.form["embarked"])

        features = np.array([[pclass, sex, age, fare, embarked]])
        features = scaler.transform(features)

        result = model.predict(features)[0]
        prediction = "Survived" if result == 1 else "Did Not Survive"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
