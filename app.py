from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model, Model

app = Flask(__name__)

# Modeli yükle
model = load_model("model.h5")

# Eğer model.input tanımlı değilse, dummy input ile çağır
if not hasattr(model, 'input') or model.input is None:
    dummy_input = np.zeros((1, 28, 28, 1))  # MNIST boyutu
    model.predict(dummy_input)  # Modeli bir kere çağır, input tensor oluşur

# Ara katman çıktılarını almak için activation modeli oluştur
conv1_output = model.layers[0].output
conv2_output = model.layers[1].output
activation_model = Model(inputs=model.input, outputs=[conv1_output, conv2_output])

@app.route('/')
def home():
    return "Model deployed successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # JSON'dan gelen veriyi uygun numpy array formatına çevir
    img = np.array(data['image']).reshape(1, 28, 28, 1)
    preds = model.predict(img)
    return jsonify({'prediction': preds.tolist()})

if __name__ == "__main__":
    # Render genellikle PORT env değişkeni ile çalışır
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
