from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load trained AI model
model = joblib.load("spam_detector_model.pkl")

# Prediction history
history = []


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        message = data.get("message", "").strip()

        if not message:
            return jsonify({
                "error": "Message is required"
            }), 400

        # AI prediction
        prediction = model.predict([message])[0]

        # Confidence
        probabilities = model.predict_proba([message])[0]
        confidence = max(probabilities) * 100

        result = {
            "message": message,
            "prediction": prediction,
            "confidence": round(confidence, 2)
        }

        # Save prediction in history
        history.insert(0, result)

        # Keep only latest 20 predictions
        if len(history) > 20:
            history.pop()

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(history)


@app.route("/stats", methods=["GET"])
def get_stats():

    total = len(history)

    spam_count = sum(
        1 for item in history
        if item["prediction"] == "spam"
    )

    ham_count = sum(
        1 for item in history
        if item["prediction"] == "ham"
    )

    return jsonify({
        "total": total,
        "spam": spam_count,
        "safe": ham_count
    })


@app.route("/clear-history", methods=["DELETE"])
def clear_history():

    history.clear()

    return jsonify({
        "message": "History cleared successfully"
    })


if __name__ == "__main__":
    app.run(debug=True)