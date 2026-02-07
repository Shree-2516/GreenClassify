import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("vegetable_classifier_model.h5")

classes = [
    'Bean','Bitter_Gourd','Bottle_Gourd','Brinjal','Broccoli',
    'Cabbage','Capsicum','Carrot','Cauliflower','Cucumber',
    'Papaya','Potato','Pumpkin','Radish','Tomato'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            img = image.load_img(image_path, target_size=(224,224))
            img_array = image.img_to_array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)
            prediction = classes[np.argmax(pred)]

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)