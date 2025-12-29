from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# --------------------------
# Modeli yükle
# --------------------------
model = load_model("model.h5")

# --------------------------
# Sequential model input'u dummy ile oluştur
# --------------------------
dummy_input = np.zeros((1, 28, 28, 1))  # MNIST boyutu
_ = model(dummy_input)  # model.input tensor'u oluşur

# Conv katmanlarını al
conv1_output = model.layers[0].output
conv2_output = model.layers[1].output

activation_model = Model(inputs=model.input, outputs=[conv1_output, conv2_output])

# --------------------------
# Helper: Feature Map -> Heatmap
# --------------------------
def get_activation_maps(img_array):
    activations = activation_model.predict(img_array)
    maps = []
    for activation in activations:
        # İlk örnekten al ve normalize et
        act = activation[0]
        act = np.mean(act, axis=-1)  # kanalları birleştir
        act -= act.min()
        act /= act.max() + 1e-6
        act = cv2.resize(act, (28, 28))
        maps.append(act)
    return maps

# --------------------------
# Routes
# --------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]  # frontend'den base64 vs.
    # Örnek: base64 → array dönüşümü yapılacak, burada sadece array varsayalım
    img = np.array(data).reshape(28, 28, 1) / 255.0
    img = np.expand_dims(img, axis=0)
    
    pred = model.predict(img)
    pred_class = int(np.argmax(pred, axis=1)[0])

    heatmaps = get_activation_maps(img)
    heatmaps_list = [hm.tolist() for hm in heatmaps]  # JSON için

    return jsonify({"prediction": pred_class, "heatmaps": heatmaps_list})

# --------------------------
# Render için port ayarı
# --------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
