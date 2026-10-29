from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model using joblib
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["sqft"]),
        float(request.form["beds"]),
        float(request.form["baths"]),
        float(request.form["age"]),
        float(request.form["lot_size"]),
        float(request.form["garage_spaces"]),
        float(request.form["distance_to_transit"]),
        float(request.form["school_rating"]),
        float(request.form["has_pool"])
    ]
    
    prediction = model.predict([features])
    output = round(prediction[0], 2)
    
    return render_template("index.html", prediction_text=f"${output:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)