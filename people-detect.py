from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import io
import os
import concurrent.futures

app = Flask(__name__)

# Load the YOLO model and class labels only once during initialization
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def detect_human(image_path):
    # Load the image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Previous code remains the same...

    # Create a blob from the image and pass it through the network
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    out_layer_indices = net.getUnconnectedOutLayers()
    if out_layer_indices.ndim == 1:
        out_layer_indices = out_layer_indices.flatten()
    output_layers = [layer_names[i - 1] for i in out_layer_indices]

    # Forward pass to get the outputs
    outputs = net.forward(output_layers)

    # Process the outputs
    human_confidence = 0
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > human_confidence:
                human_confidence = confidence

    if human_confidence > 0.5:
        # Human detected
        return True, human_confidence
    else:
        # No human detected
        return False, human_confidence

def process_multiple_files(files):
    results = []
    for file in files:
        file_path = "temp_image.jpg"
        file.save(file_path)
        human_present, confidence = detect_human(file_path)
        os.remove(file_path)
        results.append({
            'human_present': human_present,
            'percentage': f"{confidence * 100:.2f}%"
        })
    
    # Calculate the average values
    num_files = len(results)
    if num_files > 0:
        average_confidence = sum(result['human_present'] for result in results) / num_files
        average_percentage = sum(float(result['percentage'].strip('%')) for result in results) / num_files
    else:
        average_confidence = 0
        average_percentage = 0
    
    return {
        'human_present': average_confidence > 0.5,
        'percentage': f"{average_percentage:.2f}%"
    }

# Route for handling the index page logic
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        files = request.files.getlist('file')  # Get a list of uploaded files
        if not files:
            return jsonify({'error': 'No selected files'}), 400

        try:
            average_results = process_multiple_files(files)
            return jsonify(average_results)
        except Exception as e:
            # Handle any exceptions that may occur during processing
            return jsonify({'error': str(e)}), 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)