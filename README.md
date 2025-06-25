# User Verification System

A web application for user identity verification using OCR (Tesseract) for Aadhaar document processing and face verification (DeepFace with VGG-Face model). The frontend is built with HTML, CSS, and JavaScript, and the backend uses Flask. This project is designed to run locally for demo purposes.

## Features

- **OCR Processing**: Extracts Date of Birth (DOB) from Aadhaar cards (PDF or image) using Tesseract in multiple languages (English, Hindi, Tamil, Telugu, Kannada, Malayalam).
- **Face Verification**: Compares a selfie with the Aadhaar card image using DeepFace (VGG-Face model).
- **Multilingual Support**: Frontend supports translations in six Indian languages.
- **Responsive Design**: User-friendly interface with light/dark mode and drag-and-drop file upload.
- **Local Hosting**: Runs on a local Flask server for demo video recording.

## Tech Stack

- **Backend**: Python, Flask, Tesseract-OCR, DeepFace, OpenCV, PyMuPDF, pdf2image
- **Frontend**: HTML, CSS, JavaScript
- **Dependencies**: TensorFlow, NumPy, Pillow

## Setup Instructions (Local Hosting)

### Prerequisites

- Python 3.9.12
- Tesseract-OCR installed on your system
- Poppler-utils for PDF processing
- Git
- A webcam for capturing selfies
- At least 8 GB RAM (recommended for DeepFace)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/user-verification.git
   cd user-verification
   ```

2. **Set Up Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Note for DeepFace Installation**:

   - DeepFace requires TensorFlow and other heavy dependencies, which can be resource-intensive.

   - Ensure you have a compatible Python version (3.9.12 recommended).

   - If you encounter issues (e.g., `ERROR: Could not find a version that satisfies the requirement tensorflow`), try:

     ```bash
     pip install tensorflow==2.11.0
     pip install deepface==0.0.75
     ```

   - On Windows, if TensorFlow installation fails due to missing Visual C++ Build Tools, download and install them from Microsoft.

   - On Linux/macOS, ensure you have `libblas` and `liblapack` installed:

     ```bash
     # Linux
     sudo apt-get install -y libblas-dev liblapack-dev
     # macOS
     brew install libblas liblapack
     ```

   - Verify DeepFace installation:

     ```bash
     python -c "from deepface import DeepFace; print('DeepFace installed')"
     ```

   - If memory errors occur during installation, close other applications or use a system with more RAM (8 GB+ recommended).

4. **Install Tesseract-OCR**

   - **Windows**:

     - Download and install from Tesseract at UB Mannheim.

     - Add Tesseract executable to your system PATH (e.g., `C:\Program Files\Tesseract-OCR\`).

     - Update `app.py` with the correct Tesseract path if needed:

       ```python
       pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
       ```

   - **Linux**:

     ```bash
     sudo apt-get update
     sudo apt-get install -y tesseract-ocr
     ```

   - **macOS**:

     ```bash
     brew install tesseract
     ```

5. **Install Poppler**

   - **Windows**:

     - Download from Poppler for Windows.
     - Extract and add the `bin` folder to your system PATH (e.g., `C:\poppler\bin`).

   - **Linux**:

     ```bash
     sudo apt-get install -y poppler-utils
     ```

   - **macOS**:

     ```bash
     brew install poppler
     ```

6. **Download Tesseract Language Data**

   Create a `tessdata` folder in the project root and download the following files:

   ```bash
   mkdir -p tessdata/tessdata
   cd tessdata/tessdata
   for lang in eng hin tam tel kan mal; do
       wget https://github.com/tesseract-ocr/tessdata/raw/main/${lang}.traineddata
   done
   cd ../..
   ```

7. **Run the Application**

   ```bash
   python app.py
   ```

   Open your browser and navigate to `http://localhost:5000`.

### Usage

1. **Fill the Form**:
   - Enter your name, age, and DOB.
   - Select a language from the dropdown (English, Hindi, Tamil, Telugu, Kannada, Malayalam).
2. **Capture Selfie**:
   - Click "Open Camera" and capture a clear selfie.
   - Ensure good lighting and one face in the frame.
3. **Upload Aadhaar Document**:
   - Drag and drop or select an Aadhaar card (PDF or image).
4. **Submit**:
   - Click "Validate" to process the data.
   - The backend will:
     - Extract DOB from the Aadhaar document using Tesseract.
     - Verify the age and DOB match the form data.
     - Compare the selfie with the Aadhaar image using DeepFace.
   - Results are displayed with success or error messages.

### Demo Video

- A demo video will showcase:
  - Launching the local server.
  - Filling the form in multiple languages.
  - Capturing a selfie and uploading an Aadhaar document.
  - Displaying verification results.
- Record the demo using screen recording software (e.g., OBS Studio).

### Troubleshooting

- **Tesseract Not Found**: Ensure Tesseract-OCR is installed and the path is correct in `app.py`.
- **Poppler Not Found**: Add Poppler’s `bin` folder to your system PATH.
- **DeepFace Errors**:
  - Verify TensorFlow installation: `pip show tensorflow`.
  - Ensure sufficient RAM (8 GB+).
  - If DeepFace fails to load models, check the model files in `~/.deepface/weights/` and re-run `pip install deepface`.
- **Camera Access Denied**: Check browser permissions for camera access.
- **OCR Errors**: Ensure the Aadhaar document is clear and high-quality (300 DPI recommended).

### Future Improvements

- Optimize DeepFace models for faster inference.
- Add support for additional document types (e.g., passport, driver’s license).
- Deploy to a cloud platform (e.g., Heroku, AWS) for production.
- Enhance OCR accuracy with advanced preprocessing.

### 
