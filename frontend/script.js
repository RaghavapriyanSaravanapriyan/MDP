const API_URL = 'http://localhost:5000/api';

const views = {
    setup: document.getElementById('setup-view'),
    loading: document.getElementById('loading-view'),
    demo: document.getElementById('demo-view')
};

// Video Elements
const setupVideo = document.getElementById('setup-video');
const demoVideo = document.getElementById('demo-video');

// Overlay Elements
const setupOverlay = document.getElementById('setup-overlay');
const demoOverlay = document.getElementById('demo-overlay');

// Hidden capture canvas
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

let stream = null;
let recognitionInterval = null;
let setupTrackingInterval = null;

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
            
            // Sync capture canvas
            captureCanvas.width = width;
            captureCanvas.height = height;
            
            // Sync overlay canvas if provided
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
        setupStatus.textContent = "System ready. Enter name and take a few pics.";
        
        // Start a lightweight tracking loop for setup feedback
        if (setupTrackingInterval) clearInterval(setupTrackingInterval);
        setupTrackingInterval = setInterval(() => {
            // We just send to recognize but ignore identity, only show the box
            performQuickDetection(setupVideo, setupOverlay);
        }, 800);
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
    
    // Aesthetic: White border for detection
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.setLineDash([5, 5]); 
    ctx.strokeRect(x, y, w, h);
    
    // Label
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px Space Mono';
    const text = label ? `${label} (${confidence}%)` : 'DETECTING...';
    ctx.fillText(text, x, y - 10);
}

async function performQuickDetection(video, overlay) {
    if (!stream) return;
    const imgB64 = captureFrame(video);
    try {
        const res = await fetch(`${API_URL}/recognize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imgB64 })
        });
        const data = await res.json();
        drawFaceOverlay(overlay, data.box, null, 0);
    } catch(e) {}
}

shutterBtn.addEventListener('click', async () => {
    const name = personName.value.trim();
    if (!name) {
        setupStatus.textContent = "Please enter a name first.";
        return;
    }
    
    setupStatus.textContent = "Capturing...";
    const imageB64 = captureFrame(setupVideo);
    
    try {
        const res = await fetch(`${API_URL}/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, image: imageB64 })
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            setupStatus.textContent = `Captured! Take ${~~(Math.random()*2)+2} more for better accuracy.`;
        } else {
            setupStatus.textContent = `Error: ${data.message}`;
        }
    } catch (e) {
        setupStatus.textContent = "Network error.";
    }
});

uploadBtn.addEventListener('click', () => imageUpload.click());

imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

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

        try {
            const res = await fetch(`${API_URL}/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, image: imageB64 })
            });
            const data = await res.json();
            setupStatus.textContent = data.status === 'success' ? `Success!` : `Error: ${data.message}`;
            setTimeout(() => { imagePreview.style.display = 'none'; }, 2000);
        } catch (err) {
            setupStatus.textContent = "Upload failed.";
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
        progress += 1; 
        if (progress > 98) progress = 98;
        progressBar.style.width = `${progress}%`;
    }, 100);

    try {
        const res = await fetch(`${API_URL}/train`, { method: 'POST' });
        const data = await res.json();
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        
        if (data.status === 'success') {
            setTimeout(startDemo, 800);
        } else {
            alert(`Error: ${data.message}`);
            showView('setup');
        }
    } catch (e) {
        clearInterval(progressInterval);
        showView('setup');
    }
});

async function startDemo() {
    showView('demo');
    await startCamera(demoVideo, demoOverlay);
    if (recognitionInterval) clearInterval(recognitionInterval);
    recognitionInterval = setInterval(processRecognition, 1000);
}

async function processRecognition() {
    if (!stream) return;
    const imageB64 = captureFrame(demoVideo);
    
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
        console.error("Link offline");
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
