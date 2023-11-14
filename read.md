## Auto ML
It involves the automation of tasks such as feature engineering, model selection, hyperparameter tuning, and model evaluation.

#### There are many AutoML vendors like -
- AutoWEKA
- Auto-sklearn
- TPOT
- H2O AutoML
- TransmogrifAI
- MLBoX 
- Google Cloud AutoML
- Azure Automated ML

Here, we will be using TPOT.

### AutoML with TPOT for Iris Dataset Classification
```Python
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train TPOTClassifier
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42, config_dict='TPOT sparse')
tpot.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = tpot.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Export the optimized pipeline as a Python script
tpot.export('iris_pipeline.py')

# After fitting TPOTClassifier
best_pipeline = tpot.fitted_pipeline_
print("Best Pipeline:")
print(best_pipeline)

# To get the final pipeline in scikit-learn format (for example, to print the steps)
scikit_learn_pipeline = tpot.fitted_pipeline_.steps[-1][1]
print("Best Model (Scikit-learn Format):")
print(scikit_learn_pipeline)

import joblib 
joblib.dump(scikit_learn_pipeline, "model.joblib")
```
Using joblib, we save Scikit-learn pipeline to persist the trained machine learning model and associated preprocessing steps for later use or deployment.
### Applying LIME (Local Interpretable Model-Agnostic Explanations) to AutoML-generated model
The primary purpose of applying LIME to an AutoML model is to provide interpretable and human-understandable explanations for individual predictions.
```python
#!pip install lime
from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(X_train, mode="classification", training_labels=y_train)
data_point_idx = 0  # Change this to the index of the data point you want to explain
data_point = X_test[data_point_idx]
explanation = explainer.explain_instance(data_point, scikit_learn_pipeline.predict_proba)
explanation.show_in_notebook()
```
#

Further, we can deploy the saved joblib file to Hugging Face using flask.

#### app.py
```python
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
```
To run **:** python app.py

curl command **:** curl -X POST -H "Content-Type: application/json" -d "{\"features\": [[4.8, 3.0, 1.4, 0.3]]}" http://127.0.0.1:5000/predict
## Reference

https://medium.com/nerd-for-tech/what-is-automl-automated-machine-learning-a-brief-overview-a3a19c38b5f

deploy flask api to huggingface - https://www.youtube.com/watch?v=pWnE9FHnGcQ