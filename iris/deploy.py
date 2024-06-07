from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
model = pickle.load(open("saved_model1.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from form input
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]

        # Make prediction using the loaded model
        prediction = model.predict(final_features)
        output = prediction[0]

        # Render the result
        return render_template(
            "index.html", prediction_text="Predicted Class: {}".format(output)
        )
    except Exception as e:
        # In case of error, return an error message
        return render_template("index.html", prediction_text="Error: {}".format(e))


if __name__ == "__main__":
    app.run(debug=True)
