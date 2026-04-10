
from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and scalers (ensure these files exist in your project root)
try:
    model = pickle.load(open('model.pkl', 'rb'))
    mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    print("ML models loaded successfully")
except FileNotFoundError as e:
    print(f"Model file not found: {e}")
    model = mx = sc = None

# Soil type encoding dictionary
soil_mapping = {'sandy': 0, 'clay': 1, 'loamy': 2}

# Crop mapping for predictions
df_dict = {
    1:'rice', 2:'maize', 3:'jute', 4:'cotton', 5:'coconut',
    6:'papaya', 7:'orange', 8:'apple', 9:'muskmelon', 10:'watermelon',
    11:'grapes', 12:'mango', 13:'banana', 14:'pomegranate', 15:'lentil',
    16:'blackgram', 17:'mungbean', 18:'mothbeans', 19:'pigeonpeas',
    20:'kidneybeans', 21:'chickpea', 22:'coffee'
}

# AVANI Routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/language")
def language():
    return render_template("language.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/solution")
def solution():
    return render_template("solution.html")

@app.route("/recommendation")
def recommendation():
    return render_template("recommendation.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# New route for detailed crop analysis
@app.route("/crop-analysis")
def crop_analysis():
    return render_template("crop_analysis.html")

@app.route("/shop")
def shop():
    return render_template("shop.html")

@app.route("/community")
def community():
    return render_template("community.html")

# AI Recommendation API endpoint
@app.route("/recommend", methods=["POST"])
def recommend():
    if not all([model, mx, sc]):
        return jsonify({"error": "ML models not loaded properly"}), 500
    
    data = request.json
    try:
        N = float(data.get("N"))
        P = float(data.get("P"))
        K = float(data.get("K"))
        temperature = float(data.get("temperature"))
        humidity = float(data.get("humidity"))
        ph = float(data.get("ph"))
        rainfall = float(data.get("rainfall"))
        soil_type = data.get("soil_type").lower()

        if soil_type not in soil_mapping:
            return jsonify({"error": "Invalid soil type. Choose from sandy, clay, loamy."}), 400
        
        soil_type_encoded = soil_mapping[soil_type]
        
        # Prepare features for prediction
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_type_encoded]])
        features_scaled = sc.transform(mx.transform(features))
        
        # Get prediction
        prediction = model.predict(features_scaled)[0]
        crop_name = df_dict.get(prediction, "Unknown")
        
        # Get prediction probabilities for confidence
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = max(probabilities) * 100
        
        # Get top 3 recommendations
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_crops = [df_dict.get(i+1, "Unknown") for i in top_indices]
        top_confidences = [probabilities[i] * 100 for i in top_indices]
        
        return jsonify({
            "recommended_crop": crop_name,
            "confidence": round(confidence, 2),
            "top_recommendations": [
                {"crop": crop, "confidence": round(conf, 2)} 
                for crop, conf in zip(top_crops, top_confidences)
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Form submission handler for recommendation page
@app.route("/submit-recommendation", methods=["POST"])
def submit_recommendation():
    try:
        # Get form data
        location = request.form.get('location')
        last_crop = request.form.get('lastCrop')
        irrigation = request.form.get('irrigation')
        
        # Store user info and redirect to crop analysis
        # In a real app, you'd save this to a database
        return redirect(url_for('crop_analysis'))
    
    except Exception as e:
        return render_template("recommendation.html", error=str(e))

# Contact form handler
@app.route("/submit-contact", methods=["POST"])
def submit_contact():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # In a real app, you'd save this to database or send email
        print(f"Contact form submitted: {name}, {email}, {message}")
        
        return render_template("contact.html", success="Message sent successfully!")
    
    except Exception as e:
        return render_template("contact.html", error="Failed to send message.")

if __name__ == "__main__":
    app.run(debug=True)