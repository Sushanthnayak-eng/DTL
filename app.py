from flask import Flask, render_template, request, redirect, session
import joblib
import numpy as np
import json
from pathlib import Path

app = Flask(__name__)
app.secret_key = "mysupersecretkey"   # Change it for security

# -------------------------
# Load ML Model
# -------------------------
MODELS_DIR = Path("models")
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_PATH = MODELS_DIR / "compost_model.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.pkl"

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

# -------------------------
# User System (JSON File)
# -------------------------
USER_FILE = "users.json"

def load_users():
    try:
        with open(USER_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

# -------------------------
# Routes for Website Pages
# -------------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

# -------------------------
# Authentication Routes
# -------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        users = load_users()
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            session["user"] = username
            return redirect("/dashboard")
        else:
            return "<h3>Invalid username or password</h3>"

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        users = load_users()
        username = request.form["username"]
        password = request.form["password"]

        if username in users:
            return "<h3>User already exists. Try login.</h3>"
        else:
            users[username] = password
            save_users(users)
            return redirect("/login")

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# -------------------------
# Dashboard Page
# -------------------------

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")

    return render_template("dashboard.html", username=session["user"])

# -------------------------
# Suggestions Function
# -------------------------

def get_suggestions(sample):
    s = []

    if sample["pH"] < 6.5:
        s.append("pH is low — add agricultural lime or wood ash.")
    if sample["pH"] > 8:
        s.append("pH is high — add composted leaves to lower pH.")

    if sample["MC(%)"] < 20:
        s.append("Moisture too low — add water and mix properly.")
    if sample["MC(%)"] > 40:
        s.append("Moisture too high — dry the compost under sunlight.")

    if sample["C/N Ratio"] > 25:
        s.append("C/N ratio high — add nitrogen-rich waste to balance it.")

    return s if s else ["Compost quality looks good."]

# -------------------------
# Prediction Page
# -------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if "user" not in session:
        return redirect("/login")

    label, prob, suggestions = None, None, None

    if request.method == "POST":

        # Collect input values
        data = {}
        for f in feature_names:
            value = float(request.form.get(f, 0))
            data[f] = value

        # Convert to array
        arr = np.array([data[f] for f in feature_names]).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

        # Predict
        pred = model.predict(arr_scaled)[0]
        prob = float(model.predict_proba(arr_scaled)[0][1])

        label = "Suitable" if pred == 1 else "Not Suitable"

        # Suggestions
        if label == "Not Suitable":
            suggestions = get_suggestions(data)
        else:
            suggestions = ["Compost appears suitable."]

    return render_template("index.html",
                           feature_names=feature_names,
                           result=label,
                           prob=prob,
                           suggestions=suggestions)


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
