from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create directories for storing captured images
os.makedirs('captured_images', exist_ok=True)
os.makedirs('processed_images', exist_ok=True)

class ImageProcessor:
    def __init__(self):
        self.capture_count = 0
        logger.info("Image Processor initialized")
    
    def process_image(self, image_data, timestamp):
        """
        Process the captured image and return prediction results
        """
        try:
            # Convert base64 image to OpenCV format
            image_bytes = base64.b64decode(image_data)
            np_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"error": "Could not decode image"}
            
            # Save the captured image
            filename = f"captured_images/capture_{timestamp.replace(':', '-')}.jpg"
            cv2.imwrite(filename, image)
            self.capture_count += 1
            
            # Preprocess image for model input
            processed_image = self.preprocess_image(image)
            
            # Save processed image
            processed_filename = f"processed_images/processed_{timestamp.replace(':', '-')}.jpg"
            cv2.imwrite(processed_filename, processed_image)
            
            # Here you would normally pass the image to your ML model
            # For now, we'll simulate a prediction
            prediction, confidence = self.simulate_prediction()
            
            logger.info(f"Processed image {self.capture_count}, Prediction: {prediction}, Confidence: {confidence}%")
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "timestamp": timestamp,
                "capture_count": self.capture_count,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {"error": str(e)}
    
    def preprocess_image(self, image):
        """
        Preprocess the image for model input
        """
        # Resize to 200x200 (matching your dataset)
        resized = cv2.resize(image, (200, 200))
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        normalized = rgb_image.astype(np.float32) / 255.0
        
        return normalized
    
    def simulate_prediction(self):
        """
        Simulate ML model prediction (replace with actual model inference)
        """
        # Simulate ASL alphabet predictions
        asl_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                       'SPACE', 'DELETE', 'NOTHING']
        
        # Simulate realistic confidence scores
        confidence = np.random.randint(70, 95)
        
        # Occasionally return nothing to simulate real scenarios
        if np.random.random() < 0.3:
            return "NOTHING", confidence
        
        # Return a random letter from the alphabet
        prediction = np.random.choice(asl_alphabet[:-3])  # Exclude SPACE, DELETE, NOTHING
        
        return prediction, confidence

# Initialize the image processor
processor = ImageProcessor()

@app.route('/process-image', methods=['POST'])
def process_image():
    """
    Endpoint to receive and process images from the frontend
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = data['image']
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        # Process the image
        result = processor.process_image(image_data, timestamp)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in process-image endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "captures_processed": processor.capture_count,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/capture-count', methods=['GET'])
def get_capture_count():
    """Get total number of captures processed"""
    return jsonify({
        "capture_count": processor.capture_count
    })

if __name__ == '__main__':
    logger.info("Starting SignFlow Backend Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)