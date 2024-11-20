from flask import Flask, request, jsonify, send_file
import torch
import base64
import io
from PIL import Image
import numpy as np
from models import Generator, StyleTransferNet
from training_utils import DataProcessor

app = Flask(__name__)

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
style_transfer = StyleTransferNet().to(device)

# Load pretrained weights
generator.load_state_dict(torch.load('weights/generator.pth', map_location=device))
style_transfer.load_state_dict(torch.load('weights/style_transfer.pth', map_location=device))

# Initialize data processor
processor = DataProcessor()

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Get noise parameters from request
        data = request.json
        noise_param1 = float(data.get('noise_param1', 50)) / 100
        noise_param2 = float(data.get('noise_param2', 50)) / 100
        
        # Generate random noise vector influenced by parameters
        z = torch.randn(1, 100).to(device)
        z = z * noise_param1 + noise_param2
        
        # Generate image
        with torch.no_grad():
            generated = generator(z)
        
        # Convert to PIL Image
        img_data = processor.tensor_to_pil(generated[0])
        
        # Convert to base64
        buffered = io.BytesIO()
        img_data.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({'image': img_str})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/style-transfer', methods=['POST'])
def transfer_style():
    try:
        # Get uploaded image and style selection
        file = request.files['image']
        style = request.form.get('style', 'vangogh')
        
        # Process uploaded image
        image = Image.open(file.stream)
        image_tensor = processor.transform(image).unsqueeze(0).to(device)
        
        # Apply style transfer
        with torch.no_grad():
            styled = style_transfer(image_tensor)
        
        # Convert to PIL Image
        styled_image = processor.tensor_to_pil(styled[0])
        
        # Convert to base64
        buffered = io.BytesIO()
        styled_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({'image': img_str})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
