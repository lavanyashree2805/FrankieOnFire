from flask import Flask, request, jsonify, render_template
import pytesseract
import cv2
import re
from datetime import datetime
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from deepface import DeepFace
from pdf2image import convert_from_path
import os
import shutil

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
DOCUMENT_FOLDER = os.path.join(UPLOAD_FOLDER, 'documents')
SELFIE_FOLDER = os.path.join(UPLOAD_FOLDER, 'selfies')
TESSERACT_DATA = './tessdata'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
os.makedirs(SELFIE_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tesseract configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update for your system
os.environ['TESSDATA_PREFIX'] = TESSERACT_DATA

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    return resized

def extract_text(image):
    lang_string = 'eng+hin+tam+tel+kan+mal'
    return pytesseract.image_to_string(image, lang=lang_string)

def parse_aadhar_text(text):
    dob_pattern = r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b'
    dob_match = re.search(dob_pattern, text)
    dob = dob_match.group(1) if dob_match else None
    return None, dob

def calculate_age(dob_str):
    formats = ['%d/%m/%Y', '%d-%m-%Y']
    for fmt in formats:
        try:
            dob = datetime.strptime(dob_str, fmt)
            today = datetime.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return age, age >= 18
        except:
            continue
    return None, None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    try:
        # Validate form data
        name = request.form.get('name')
        age = request.form.get('age')
        dob = request.form.get('dob')
        if 'image' not in request.files or 'document' not in request.files:
            return jsonify({'error': 'Missing image or document'}), 400

        image_file = request.files['image']
        doc_file = request.files['document']

        if not (image_file and allowed_file(image_file.filename)) or not (doc_file and allowed_file(doc_file.filename)):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save files
        image_path = os.path.join(SELFIE_FOLDER, image_file.filename)
        doc_path = os.path.join(DOCUMENT_FOLDER, doc_file.filename)
        image_file.save(image_path)
        doc_file.save(doc_path)

        # Process document
        if doc_path.lower().endswith('.pdf'):
            images = convert_from_path(doc_path, dpi=300)
            aadhaar_image_path = os.path.join(DOCUMENT_FOLDER, 'aadhaar_page1.png')
            images[0].save(aadhaar_image_path, 'PNG')
        else:
            aadhaar_image_path = doc_path

        # OCR processing
        image = Image.open(aadhaar_image_path)
        processed_image = preprocess_image(image)
        text = extract_text(processed_image)
        _, extracted_dob = parse_aadhar_text(text)

        # Validate DOB and age
        if extracted_dob:
            calculated_age, is_adult = calculate_age(extracted_dob)
            if calculated_age is None or str(calculated_age) != age or extracted_dob != dob:
                return jsonify({'error': 'DOB or age mismatch', 'dob': extracted_dob, 'age': calculated_age}), 400
        else:
            return jsonify({'error': 'DOB not found in document'}), 400

        # Face verification
        try:
            result = DeepFace.verify(
                img1_path=aadhaar_image_path,
                img2_path=image_path,
                model_name='VGG-Face',
                enforce_detection=False
            )
            similarity_score = 100 - (result['distance'] * 100)
            threshold = 50.0

            if similarity_score < threshold:
                return jsonify({'error': 'Faces do not match', 'score': f'{similarity_score:.2f}%'}), 400

            # Clean up uploaded files
            for folder in [DOCUMENT_FOLDER, SELFIE_FOLDER]:
                shutil.rmtree(folder)
                os.makedirs(folder)

            return jsonify({
                'message': 'Verification successful',
                'dob': extracted_dob,
                'age': calculated_age,
                'is_adult': is_adult,
                'face_match_score': f'{similarity_score:.2f}%'
            })

        except Exception as e:
            return jsonify({'error': f'Face verification failed: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)