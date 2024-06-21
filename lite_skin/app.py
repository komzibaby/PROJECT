from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load your pre-trained skin cancer detection model
model = tf.keras.models.load_model('C:\\Users\\HP\\Downloads\\lite_skin\\model\skinguard.h5')

# Define the class names
class_names = [
    'Actinic keratoses and intraepithelial carcinomae', 
    'basal cell carcinoma',
    'benign keratosis-like lesions', 
    'dermatofibroma', 
    'Melanocytic nevi',
    'pyogenic granulomas and hemorrhage', 
    'Melanoma'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    try:
        # Read the image
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((28, 28))  # Resize image to match model input
        img_array = np.array(image).reshape(-1, 28, 28, 3) / 255.0  # Normalize the image
        
        # Predict using the model
        predictions = model.predict(img_array)
        max_prob = np.max(predictions) * 100
        class_index = np.argmax(predictions)
        predicted_class = class_names[class_index]

        return jsonify({
            'prediction': predicted_class,
            'confidence': round(max_prob, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
