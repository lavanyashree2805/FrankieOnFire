import streamlit as st
from PIL import Image
import numpy as np
import pytesseract
import cv2
import re
from datetime import datetime
from pdf2image import convert_from_path
from deepface import DeepFace
import os
import io

# Supported OCR languages
languages = ['eng', 'hin', 'tam', 'tel', 'kan', 'mal']

# === Helper Functions ===
def preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    return resized

def extract_text(image):
    lang_string = '+'.join(languages)
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

# === Streamlit App ===
st.title("Aadhaar Age & Identity Verification")

st.header("Step 1: Upload Aadhaar Card (PDF or Image)")
aadhaar_file = st.file_uploader("Upload Aadhaar card", type=["pdf", "png", "jpg", "jpeg"])

if aadhaar_file is not None:
    if aadhaar_file.name.lower().endswith(".pdf"):
        images = convert_from_path(aadhaar_file, dpi=300)
        aadhaar_image = images[0]
    else:
        aadhaar_image = Image.open(aadhaar_file)

    processed_image = preprocess_image(aadhaar_image)
    text = extract_text(processed_image)
    _, dob = parse_aadhar_text(text)

    if dob:
        age, is_adult = calculate_age(dob)
    else:
        age, is_adult = None, None

    st.subheader("Extracted Information")
    st.write(f"**DOB:** {dob if dob else 'Not found'}")
    st.write(f"**Age:** {age if age is not None else 'N/A'}")
    st.write(f"**18+:** {'Yes' if is_adult else 'No'}")

    st.header("Step 2: Upload Selfie")
    selfie_file = st.file_uploader("Upload a selfie image", type=["jpg", "jpeg", "png"])

    if selfie_file is not None:
        with open("aadhaar_image_temp.png", "wb") as f:
            aadhaar_image.save(f)
        with open("selfie_temp.png", "wb") as f:
            f.write(selfie_file.read())

        try:
            result = DeepFace.verify(
                img1_path="aadhaar_image_temp.png",
                img2_path="selfie_temp.png",
                model_name='VGG-Face',
                enforce_detection=False
            )
            similarity_score = 100 - (result['distance'] * 100)
            st.subheader("Face Match Result")
            st.write(f"**Confidence Score:** {similarity_score:.2f}%")
            if similarity_score >= 50.0:
                st.success("✅ Faces Matched")
            else:
                st.error("❌ Faces Do Not Match")

        except Exception as e:
            st.error(f"Face verification error: {str(e)}")
