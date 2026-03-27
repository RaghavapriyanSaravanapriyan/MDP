const API_URL = 'http://localhost:5000/api';

const views = {
    setup: document.getElementById('setup-view'),
    loading: document.getElementById('loading-view'),
    demo: document.getElementById('demo-view')
};

const setupVideo = document.getElementById('setup-video');
const demoVideo = document.getElementById('demo-video');
const setupOverlay = document.getElementById('setup-overlay');
const demoOverlay = document.getElementById('demo-overlay');
const captureCanvas = document.getElementById('canvas');
const captureCtx = captureCanvas.getContext('2d');

const shutterBtn = document.getElementById('shutter-btn');
const uploadBtn = document.getElementById('upload-btn');
const imageUpload = document.getElementById('image-upload');
const imagePreview = document.getElementById('image-preview');
const personName = document.getElementById('person-name');
const setupStatus = document.getElementById('setup-status');
const trainBtn = document.getElementById('train-btn');
const restartBtn = document.getElementById('restart-btn');

const progressBar = document.getElementById('progress');
const lastPerson = document.getElementById('last-person');
const lockStatus = document.getElementById('lock-status');
const teamMembersList = document.getElementById('team-members');

const captureCountDisplay = document.getElementById('capture-count');

let stream = null;
let recognitionInterval = null;
let setupTrackingInterval = null;
let isProcessing = false;
let captureCount = 0;

async function init() {
    await resetSystem();
    loadTeamMembers();
}

async function startCamera(videoElement, overlayCanvas) {
    try {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        videoElement.srcObject = stream;
        
        videoElement.onloadedmetadata = () => {
            const width = videoElement.videoWidth;
            const height = videoElement.videoHeight;
            captureCanvas.width = width;
            captureCanvas.height = height;
            if (overlayCanvas) {
                overlayCanvas.width = width;
                overlayCanvas.height = height;
            }
        };
    } catch (err) {
        console.error("Camera error:", err);
        setupStatus.textContent = "Error accessing camera.";
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    if (setupTrackingInterval) clearInterval(setupTrackingInterval);
    if (recognitionInterval) clearInterval(recognitionInterval);
}

function showView(viewName) {
    Object.values(views).forEach(v => v.classList.remove('active'));
    views[viewName].classList.add('active');
}

async function resetSystem() {
    try {
        await fetch(`${API_URL}/setup`, { method: 'POST' });
        showView('setup');
        await startCamera(setupVideo, setupOverlay);
        personName.value = '';
        captureCount = 0;
        setupStatus.textContent = "Ready. Enter name and click shutter.";
        
        if (setupTrackingInterval) clearInterval(setupTrackingInterval);
        setupTrackingInterval = setInterval(() => {
            performQuickDetection(setupVideo, setupOverlay);
        }, 1200); // Efficient tracking frequency
        updateCaptureUI();
    } catch (e) {
        console.error("Reset failed", e);
    }
}

function captureFrame(videoElement) {
    captureCtx.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
    return captureCanvas.toDataURL('image/jpeg', 0.8);
}

function drawFaceOverlay(canvas, box, label, confidence) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!box) return;
    
    const [x, y, w, h] = box;
    ctx.strokeStyle = label ? '#00ff00' : '#ffffff';
    ctx.lineWidth = 3;
    ctx.setLineDash(label ? [] : [5, 5]); 
    ctx.strokeRect(x, y, w, h);
    
    if (label) {
        ctx.fillStyle = '#00ff00';
        ctx.font = 'bold 16px Space Mono';
        ctx.fillText(`${label} (${confidence}%)`, x, y - 10);
    }
}

async function performQuickDetection(video, overlay) {
    if (!stream || isProcessing) return;
    const imgB64 = captureFrame(video);
    isProcessing = true;
    try {
        const res = await fetch(`${API_URL}/recognize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imgB64, fast_only: true })
        });
        const data = await res.json();
        drawFaceOverlay(overlay, data.box, null, 0);
    } catch(e) {
    } finally {
        isProcessing = false;
    }
}

function updateCaptureUI() {
    captureCountDisplay.textContent = captureCount;
    trainBtn.disabled = captureCount < 5;
    if (captureCount >= 5) {
        setupStatus.textContent = "Requirement met. You can start training or capture more.";
    }
}

shutterBtn.addEventListener('click', async () => {
    const name = personName.value.trim();
    if (!name || isProcessing) return;
    
    isProcessing = true;
    setupStatus.textContent = "Processing capture...";
    const imageB64 = captureFrame(setupVideo);
    
    try {
        const res = await fetch(`${API_URL}/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, image: imageB64 })
        });
        const data = await res.json();
        if (data.status === 'success') {
            captureCount++;
            setupStatus.textContent = `Captured! (Total: ${captureCount})`;
            updateCaptureUI();
        } else {
            setupStatus.textContent = `Error: ${data.message}`;
        }
    } catch (e) {
        setupStatus.textContent = "Network error. Server might be busy.";
    } finally {
        isProcessing = false;
    }
});

uploadBtn.addEventListener('click', () => imageUpload.click());

imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file || isProcessing) return;

    const name = personName.value.trim();
    if (!name) {
        setupStatus.textContent = "Enter name first.";
        imageUpload.value = '';
        return;
    }

    const reader = new FileReader();
    reader.onload = async (event) => {
        const imageB64 = event.target.result;
        imagePreview.style.backgroundImage = `url(${imageB64})`;
        imagePreview.style.display = 'block';
        setupStatus.textContent = "Uploading...";
        isProcessing = true;

        try {
            const res = await fetch(`${API_URL}/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, image: imageB64 })
            });
            const data = await res.json();
            setupStatus.textContent = data.status === 'success' ? `Uploaded!` : `Error: ${data.message}`;
            if (data.status === 'success') {
                captureCount++;
                updateCaptureUI();
            }
            setTimeout(() => { imagePreview.style.display = 'none'; }, 2000);
        } catch (err) {
            setupStatus.textContent = "Upload failed.";
        } finally {
            isProcessing = false;
        }
    };
    reader.readAsDataURL(file);
});

trainBtn.addEventListener('click', async () => {
    if (setupTrackingInterval) clearInterval(setupTrackingInterval);
    showView('loading');
    stopCamera();
    
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 2; 
        if (progress > 98) progress = 98;
        progressBar.style.width = `${progress}%`;
    }, 250);

    try {
        const res = await fetch(`${API_URL}/train`, { method: 'POST' });
        const data = await res.json();
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        if (data.status === 'success') {
            setTimeout(startDemo, 1000);
        } else {
            alert(`Indexing Error: ${data.message}. Ensure you captured enough photos.`);
            showView('setup');
        }
    } catch (e) {
        clearInterval(progressInterval);
        alert("Server timeout during indexing. Model initialization can take 1-2 mins on first run.");
        showView('setup');
    }
});

async function startDemo() {
    showView('demo');
    await startCamera(demoVideo, demoOverlay);
    if (recognitionInterval) clearInterval(recognitionInterval);
    recognitionInterval = setInterval(processRecognition, 2000); // Optimized for accuracy vs CPU
}

async function processRecognition() {
    if (!stream || isProcessing) return;
    const imageB64 = captureFrame(demoVideo);
    isProcessing = true;
    
    try {
        const res = await fetch(`${API_URL}/recognize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageB64 })
        });
        const data = await res.json();
        if (data.status === 'success') {
            drawFaceOverlay(demoOverlay, data.box, data.name, data.confidence);
            if (data.name !== 'Unknown') {
                lastPerson.textContent = `${data.name} (${data.confidence}%)`;
            }
            updateLock(data.name);
        }
    } catch (e) {
        console.error("Connection lost");
    } finally {
        isProcessing = false;
    }
}

function updateLock(name) {
    const isLocked = name === 'Unknown';
    const h1 = lockStatus.querySelector('h1');
    lockStatus.className = isLocked ? 'lock-status locked' : 'lock-status unlocked';
    h1.textContent = isLocked ? 'LOCKED' : 'UNLOCKED';
}

restartBtn.addEventListener('click', () => {
    stopCamera();
    resetSystem();
});

init();
