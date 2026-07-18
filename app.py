from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

print("Model Loaded Successfully")


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    # Get form values
    age = float(request.form["age"])
    sex = request.form["sex"]
    dataset = request.form["dataset"]
    cp = request.form["cp"]
    trestbps = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    fbs = request.form["fbs"]
    restecg = request.form["restecg"]
    thalch = float(request.form["thalch"])
    exang = request.form["exang"]
    oldpeak = float(request.form["oldpeak"])
    slope = request.form["slope"]

    # Create DataFrame
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "dataset": dataset,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Prediction probability
    probabilities = model.predict_proba(input_df)[0]

    # Confidence percentage
    confidence = probabilities[prediction] * 100

    # Result message
    if prediction == 1:
        result = "❤️ High Risk of Heart Disease"
        advice = (
            "The entered health parameters indicate an elevated risk. "
            "Please consult a healthcare professional for proper medical evaluation."
        )
    else:
        result = "💚 Low Risk of Heart Disease"
        advice = (
            "The entered health parameters indicate a lower risk. "
            "Continue maintaining a healthy lifestyle and regular health check-ups."
        )

    return render_template(
        "index.html",
        prediction_text=result,
        confidence=round(confidence, 2),
        advice=advice
    )


if __name__ == "__main__":
    app.run(debug=True)