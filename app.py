from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import cv2
import base64

app = Flask(__name__)

# Modeli yükle
model = load_model("model.h5")

# Conv katmanları
conv1_output = model.get_layer(index=0).output
conv2_output = model.get_layer(index=2).output
activation_model = Model(inputs=model.input, outputs=[conv1_output, conv2_output])

# Grad-CAM modeli
last_conv_layer = model.get_layer(index=2)
grad_model = Model([model.inputs], [last_conv_layer.output, model.output])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    image_data = base64.b64decode(data.split(",")[1])

    img = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img_input = img.reshape(1,28,28,1)

    # Tahmin
    preds = model.predict(img_input)
    digit = int(np.argmax(preds))

    # Feature map
    activations = activation_model.predict(img_input)
    feature_maps = []
    layer_names = ["Conv1", "Conv2"]
    for l_idx, layer_activation in enumerate(activations):
        maps_for_layer = []
        for i in range(min(6, layer_activation.shape[-1])):
            fm = layer_activation[0,:,:,i]
            fm -= fm.min()
            fm /= (fm.max()+1e-5)
            fm = np.uint8(fm*255)
            _, buffer = cv2.imencode(".png", fm)
            maps_for_layer.append(base64.b64encode(buffer).decode("utf-8"))
        feature_maps.append({"layer": layer_names[l_idx], "maps": maps_for_layer})

    # Grad-CAM
    with tf.GradientTape() as tape:
        last_conv_out, pred = grad_model(img_input)
        class_idx = np.argmax(pred)
        loss = pred[:, class_idx]
    grads = tape.gradient(loss, last_conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    last_conv_out = last_conv_out[0]
    for i in range(last_conv_out.shape[-1]):
        last_conv_out[:,:,i] *= pooled_grads[i]
    heatmap = np.mean(last_conv_out, axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= (heatmap.max()+1e-5)
    heatmap = cv2.resize(heatmap.numpy(), (28,28))
    heatmap = np.uint8(255*heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    _, buffer = cv2.imencode(".png", heatmap_color)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "digit": digit,
        "feature_maps": feature_maps,
        "heatmap": heatmap_base64
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
