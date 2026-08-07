from flask import Flask, render_template, request
from predict_crop import predict_crop
from predict_disease import predict_disease
from crop_info import crop_info
from disease_info import disease_info
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/crop")
def crop():
    return render_template("crop.html")

@app.route("/predict_crop", methods=["POST"])
def predict_crop_page():

    n = float(request.form["n"])
    p = float(request.form["p"])
    k = float(request.form["k"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    # AI Prediction
    crop = predict_crop(
        n,
        p,
        k,
        temperature,
        humidity,
        ph,
        rainfall
    )
    print("predicted crop:", crop)
    print("type:",type(crop))

    # Get crop information
    crop = str(crop).strip().title()
    info = crop_info.get(crop)

    if info is None:

        info = {

            "category": "Not Available",

            "scientific_name": "Not Available",

            "season": "Not Available",

            "duration": "Not Available",

            "soil": "Not Available",

            "ideal_temperature": "Not Available",

            "water": "Not Available",

            "rainfall_need": "Not Available",

            "ideal_ph": "Not Available",

            "cultivation": [],

            "fertilizer": [],

            "care_tips": [],

            "practices": [],

            "harvest_time": "Not Available",

            "yield": "Not Available",

            "harvest_method": "Not Available",

            "market": "Not Available",

            "uses": [],

            "fact": "Information not available.",

            "diseases": []

        }

    return render_template(

        "crop_result.html",

        crop=crop,

        category=info["category"],

        scientific_name=info["scientific_name"],

        season=info["season"],

        duration=info["duration"],

        soil=info["soil"],

        ideal_temperature=info["ideal_temperature"],

        water=info["water"],

        rainfall_need=info["rainfall_need"],

        ideal_ph=info["ideal_ph"],

        cultivation=info["cultivation"],

        fertilizer=info["fertilizer"],

        care_tips=info["care_tips"],

        practices=info["practices"],

        harvest_time=info["harvest_time"],

        yield_value=info["yield"],

        harvest_method=info["harvest_method"],

        market=info["market"],

        uses=info["uses"],

        fact=info["fact"],

        diseases=info["diseases"]

    )

# ==========================================
# DISEASE DETECTION
# ==========================================

@app.route("/disease")
def disease():
    return render_template("disease.html")


@app.route("/predict_disease", methods=["POST"])
def predict_disease_page():

    image = request.files["image"]

    filename = secure_filename(image.filename)

    upload_folder = "static/uploads"

    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, filename)

    image.save(filepath)

    disease = predict_disease(filepath)
    print("Predicted disease:", disease)

    # ------------------------------------------
    # Clean disease name
    # ------------------------------------------

    disease = disease.replace("___", " ")
    disease = disease.replace("_", " ")
    disease = " ".join(disease.split())

    # Normalize common model outputs

    disease = disease.replace("healthy", "Healthy")
    disease = disease.replace("Bacterial spot", "Bacterial Spot")
    disease = disease.replace("Leaf scorch", "Leaf Scorch")
    disease = disease.replace("Common rust", "Common Rust")
    disease = disease.replace("Northern Leaf Blight", "Northern Leaf Blight")

    print("Formatted disease:", disease)

    # ------------------------------------------
    # Search disease dictionary
    # ------------------------------------------

    info = disease_info.get(disease)

    if info is None:

        for key, value in disease_info.items():

            if key.lower() == disease.lower():

                info = value
                disease = key
                break

    # ------------------------------------------
    # Default information
    # ------------------------------------------

    if info is None:

        info = {

            "symptoms": ["Information not available."],
            "causes": ["Information not available."],
            "treatment": ["Information not available."],
            "prevention": ["Information not available."],
            "care_tips": ["Information not available."],
            "fact": "Information not available."

        }

    return render_template(

        "disease_result.html",

        disease=disease,

        image_path="/" + filepath.replace("\\", "/"),

        symptoms=info["symptoms"],
        

        causes=info["causes"],

        treatment=info["treatment"],

        prevention=info["prevention"],

        care_tips=info["care_tips"],

        fact=info["fact"]

    )
# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )