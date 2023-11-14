from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the scikit-learn model
model = joblib.load("sklearn_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        features = data["features"]

        # Make predictions using the scikit-learn model
        predictions = model.predict(features)

        # Prepare the response
        response = {"predictions": predictions.tolist()}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

