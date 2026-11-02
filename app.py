import flask
from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# LOAD TRAINED IRIS MODEL
# ============================================================

with open("Iris_Model_Naive_Bayes.pkl", "rb") as file:
    model = pickle.load(file)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def main_page():

    return render_template("index.html")


# ============================================================
# IRIS PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # RECEIVE FOUR IRIS FEATURES FROM HTML
        # ----------------------------------------------------

        sepal_length = float(
            request.form["sepal_length"]
        )

        sepal_width = float(
            request.form["sepal_width"]
        )

        petal_length = float(
            request.form["petal_length"]
        )

        petal_width = float(
            request.form["petal_width"]
        )


        # ----------------------------------------------------
        # CREATE INPUT ARRAY
        # ----------------------------------------------------

        features = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(features)[0]


        # ----------------------------------------------------
        # CONVERT NUMERIC PREDICTION TO IRIS NAME
        # ----------------------------------------------------

        iris_mapping = {
            0: "Iris Setosa",
            1: "Iris Versicolor",
            2: "Iris Virginica"
        }

        prediction = iris_mapping.get(
            int(prediction),
            str(prediction)
        )


        # ----------------------------------------------------
        # SEND RESULT TO HTML
        # ----------------------------------------------------

        return render_template(
            "index.html",
            prediction_text=prediction
        )


    except Exception as e:

        return render_template(
            "index.html",
            prediction_text=f"Prediction Error: {str(e)}"
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)
