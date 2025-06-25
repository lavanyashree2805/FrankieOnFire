const form = document.getElementById('verify-form');
const resultDiv = document.getElementById('result');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const capturedImg = document.getElementById('captured-img');
const cameraBtn = document.getElementById('camera-btn');
const captureBtn = document.getElementById('capture-btn');
const retakeBtn = document.getElementById('retake-btn');
const fileInput = document.getElementById('document');
const dropZone = document.getElementById('drop-zone');
const filePreview = document.getElementById('file-preview');
const langSelect = document.getElementById('lang-select');


async function loadFaceApi() {
  await faceapi.nets.tinyFaceDetector.loadFromUri('https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights');
}
loadFaceApi();


function translatePage(lang) {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (translations[lang][key]) el.textContent = translations[lang][key];
  });

  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.getAttribute('data-i18n-placeholder');
    if (translations[lang][key]) el.placeholder = translations[lang][key];
  });

  const dropText = dropZone.querySelector('p');
  if (translations[lang]['drop_file']) dropText.textContent = translations[lang]['drop_file'];
}

langSelect.addEventListener('change', () => {
  const selectedLang = langSelect.value;
  translatePage(selectedLang);
});

translatePage('eng');


function showToast(messageKey, type = 'success') {
  const selectedLang = langSelect.value;
  const message = translations[selectedLang][messageKey] || messageKey;
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = `toast show ${type}`;
  setTimeout(() => toast.classList.remove('show'), 3000);
}


function isImageBlurred(canvas) {
  const context = canvas.getContext('2d');
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  let sum = 0, sumSquared = 0;
  for (let i = 0; i < imageData.data.length; i += 4) {
    const gray = (imageData.data[i] + imageData.data[i+1] + imageData.data[i+2]) / 3;
    sum += gray;
    sumSquared += gray * gray;
  }
  const pixelCount = imageData.data.length / 4;
  const mean = sum / pixelCount;
  const variance = (sumSquared / pixelCount) - (mean * mean);
  return variance < 100;
}


cameraBtn.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.display = 'block';
    captureBtn.style.display = 'block';
    retakeBtn.style.display = 'none';
    capturedImg.style.display = 'none';
    resultDiv.textContent = '';
  } catch {
    showToast('toast_camera_fail', 'error');
  }
};


captureBtn.onclick = async () => {
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

  if (isImageBlurred(canvas)) {
    showToast('toast_blurry_image', 'error');
    return;
  }

  const detections = await faceapi.detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions());
  if (detections.length !== 1) {
    showToast('toast_no_face', 'error');
    return;
  }

  const imageData = canvas.toDataURL('image/png');
  if (!imageData || imageData === 'data:,') {
    showToast('toast_capture_fail', 'error');
    return;
  }

  capturedImg.src = imageData;
  capturedImg.style.display = 'block';
  video.style.display = 'none';
  captureBtn.style.display = 'none';
  retakeBtn.style.display = 'block';
  video.srcObject.getTracks().forEach(track => track.stop());

  showToast('toast_capture_success', 'success');
};


retakeBtn.onclick = cameraBtn.onclick;


dropZone.onclick = () => fileInput.click();
dropZone.ondragover = (e) => {
  e.preventDefault();
  dropZone.style.background = '#d0ebff';
};
dropZone.ondragleave = () => dropZone.style.background = '#f8f9fa';
dropZone.ondrop = (e) => {
  e.preventDefault();
  fileInput.files = e.dataTransfer.files;
  updatePreview(fileInput.files[0]);
};
fileInput.onchange = () => updatePreview(fileInput.files[0]);

function updatePreview(file) {
  if (!file) return;
  if (file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = () => {
      filePreview.innerHTML = `<img src="${reader.result}" style="max-width: 100%; max-height: 200px;">`;
    };
    reader.readAsDataURL(file);
  } else {
    filePreview.textContent = `Uploaded: ${file.name}`;
  }
}


document.getElementById('theme-toggle').onclick = () => {
  document.body.classList.toggle('dark');
  const modeKey = document.body.classList.contains('dark') ? 'toast_dark_mode' : 'toast_light_mode';
  showToast(modeKey, 'success');
};


function validateForm() {
  const name = document.getElementById('name').value.trim();
  const age = parseInt(document.getElementById('age').value.trim());
  const dob = document.getElementById('dob').value;
  const image = capturedImg.src;
  const doc = fileInput.files[0];

  if (!name || !/^[A-Za-z\s]+$/.test(name)) {
    showToast('toast_invalid_name', 'error');
    return false;
  }
  if (!age || age < 1 || age > 120) {
    showToast('toast_invalid_age', 'error');
    return false;
  }
  if (!dob) {
    showToast('toast_select_dob', 'error');
    return false;
  }
  if (!image || !image.startsWith('data:image')) {
    showToast('toast_capture_first', 'error');
    return false;
  }
  if (!doc) {
    showToast('toast_upload_doc', 'error');
    return false;
  }

  const allowed = ['image/jpeg', 'image/png', 'application/pdf'];
  if (!allowed.includes(doc.type)) {
    showToast('toast_invalid_file', 'error');
    return false;
  }

  return true;
}


form.onsubmit = async (e) => {
  e.preventDefault();
  if (!validateForm()) return;

  const formData = new FormData();
  formData.append('name', document.getElementById('name').value);
  formData.append('age', document.getElementById('age').value);
  formData.append('dob', document.getElementById('dob').value);
  formData.append('image', dataURLtoBlob(capturedImg.src), 'captured-image.png');
  formData.append('document', fileInput.files[0]);

  try {
    const response = await fetch('/verify', {
      method: 'POST',
      body: formData
    });
    const result = await response.json();

    if (response.ok) {
      showToast('toast_form_success', 'success');
      resultDiv.textContent = `${translations[langSelect.value]['toast_form_success']} DOB: ${result.dob}, Age: ${result.age}, Face Match: ${result.face_match_score}`;
      resultDiv.className = 'green';
    } else {
      showToast('toast_form_error', 'error');
      resultDiv.textContent = result.error;
      resultDiv.className = 'red';
    }
  } catch (err) {
    showToast('toast_form_error', 'error');
    resultDiv.textContent = translations[langSelect.value]['toast_form_error'];
    resultDiv.className = 'red';
  }
};


function dataURLtoBlob(dataURL) {
  const [header, base64] = dataURL.split(',');
  const mime = header.match(/:(.*?);/)[1];
  const binary = atob(base64);
  const array = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) array[i] = binary.charCodeAt(i);
  return new Blob([array], { type: mime });
}