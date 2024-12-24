from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Load your trained model here
cnn_model = load_model('PerfectModel.h5')

# Manually defined class labels
class_labels = {
    0: 'Barren',
    1: 'Vegetation',
    2: 'Water',
}

def preprocess_patch(patch):
    patch = patch.resize((256, 256))  # Resize to match the model input size
    patch_array = image.img_to_array(patch)
    patch_array = patch_array / 255.0  # Normalize to [0, 1]
    patch_array = np.expand_dims(patch_array, axis=0)  # Add batch dimension
    return patch_array

def highlight_regions(img_path, patch_size=64):
    img = Image.open(img_path).convert("RGB")
    width, height = img.size

    # Prepare the mask (initialize with black)
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    class_counts = {0: 0, 1: 0, 2: 0}  # Counts for Barren, Vegetation, Water
    total_patches = 0

    # Loop through the image in patches
    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            left, upper = i, j
            right, lower = left + patch_size, upper + patch_size
            
            # Ensure we don't go out of bounds
            if right > width or lower > height:
                continue
            
            patch = img.crop((left, upper, right, lower))
            patch_array = preprocess_patch(patch)
            
            # Make predictions for the patch
            predictions = cnn_model.predict(patch_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]

            # Increment count for the predicted class
            class_counts[predicted_class_index] += 1
            total_patches += 1

            # Define colors for each class
            colors = {
                0: (255, 0, 0),    # Barren: Red
                1: (255, 255, 0),  # Vegetation: Yellow
                2: (0, 0, 255)     # Water: Blue
            }

            # Fill the mask with the corresponding color
            mask[upper:lower, left:right] = colors[predicted_class_index]

    # Create an image from the mask
    mask_image = Image.fromarray(mask)

    # Overlay the mask on the original image
    highlighted_image = Image.blend(img.convert("RGBA"), mask_image.convert("RGBA"), alpha=0.5)

    # Save highlighted image
    highlighted_image_path = os.path.join(app.config['STATIC_FOLDER'], 'highlighted_image.png')
    highlighted_image.save(highlighted_image_path)

    # Calculate percentages for each class
    percentages = {class_labels[i]: (class_counts[i] / total_patches) * 100 for i in range(3)}
    
    return highlighted_image_path, percentages

@app.route('/', methods=['GET', 'POST'])
def index():
    highlighted_image_path = None
    land_cover_percentages = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the image and make predictions
            highlighted_image_path, land_cover_percentages = highlight_regions(file_path, patch_size=16)

    return render_template('index.html', highlighted_image_path=highlighted_image_path, land_cover_percentages=land_cover_percentages)

if __name__ == '__main__':
    app.run(debug=True)
