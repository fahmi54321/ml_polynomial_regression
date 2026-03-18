# app.py

from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))
poly_reg = pickle.load(open('poly.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        level = float(data['position_level'])

        # Predict
        level_array = np.array([[level]])
        level_poly = poly_reg.transform(level_array)
        prediction = model.predict(level_poly)[0]

        # 🔵 Curve (smooth polynomial line)
        x_grid = np.arange(1, 10, 0.1).reshape(-1, 1)
        y_pred = model.predict(poly_reg.transform(x_grid))

        curve = [
            {"x": float(x), "y": float(y)}
            for x, y in zip(x_grid.flatten(), y_pred)
        ]

        # 🔴 Real dataset (bisa dari DB / CSV)
        real_data = [
            {"x": 1, "y": 45000},
            {"x": 2, "y": 50000},
            {"x": 3, "y": 60000},
            {"x": 4, "y": 80000},
            {"x": 5, "y": 110000},
            {"x": 6, "y": 150000},
            {"x": 7, "y": 200000},
            {"x": 8, "y": 300000},
            {"x": 9, "y": 500000},
            {"x": 10, "y": 1000000},
        ]

        # 🟢 User point
        user_point = {
            "x": level,
            "y": float(prediction)
        }

        return jsonify({
            # 🔥 DESKRIPTIF (tetap)
            "input": {
                "position_level": level
            },
            "prediction": {
                "salary": round(float(prediction), 2),
                "currency": "USD",
                "formatted": f"${int(prediction):,}"
            },
            "insight": {
                "category": get_salary_category(prediction),
                "confidence_note": get_confidence_note(level, prediction),
                "recommendation": get_recommendation(prediction)
            },
            "meta": {
                "model": "Polynomial Regression",
                "degree": 4
            },

            # 📊 TAMBAHAN UNTUK FLUTTER CHART
            "visualization": {
                "real_data": real_data,
                "curve": curve,
                "user_point": user_point
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# 🔍 Helper Functions

def get_salary_category(salary):
    if salary < 100000:
        return "Low to Mid Level"
    elif salary < 300000:
        return "Mid to Senior Level"
    else:
        return "Executive Level"


def get_confidence_note(level, salary):
    return f"Predicted salary for level {level} is considered realistic based on historical data distribution."


def get_recommendation(salary):
    if salary < 100000:
        return "Consider standard compensation package."
    elif salary < 300000:
        return "Competitive offer recommended."
    else:
        return "High-level negotiation required."


if __name__ == '__main__':
    app.run(debug=True)