const API_URL = 'http://localhost:5000/api';

const views = {
    setup: document.getElementById('setup-view'),
    loading: document.getElementById('loading-view'),
    demo: document.getElementById('demo-view')
};

const setupVideo = document.getElementById('setup-video');
const demoVideo = document.getElementById('demo-video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

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

async function init() {
    await resetSystem();
    loadTeamMembers();
}

async function startCamera(videoElement) {
    try {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        videoElement.onloadedmetadata = () => {
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            // Mirror canvas conditionally when drawing!
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
}

function showView(viewName) {
    Object.values(views).forEach(v => v.classList.remove('active'));
    views[viewName].classList.add('active');
}

async function resetSystem() {
    try {
        await fetch(`${API_URL}/setup`, { method: 'POST' });
        showView('setup');
        startCamera(setupVideo);
        personName.value = '';
        setupStatus.textContent = "System ready. Enter name and take a few pics (click shutter).";
    } catch (e) {
        console.error("Reset failed", e);
    }
}

function captureFrame(videoElement) {
    // We visually mirror the video in CSS, but the raw capture is not mirrored.
    // If we want mirroring on capture, we'd do ctx.translate/scale, but OpenCV doesn't care.
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
}

shutterBtn.addEventListener('click', async () => {
    const name = personName.value.trim();
    if (!name) {
        setupStatus.textContent = "Please enter a name first.";
        return;
    }
    
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
            setupStatus.textContent = `Captured and saved! (Tip: take multiple snaps of ${name})`;
        } else {
            setupStatus.textContent = `Error: ${data.message}. Make sure your face is visible.`;
        }
    } catch (e) {
        setupStatus.textContent = "Network error while saving.";
    }
});

uploadBtn.addEventListener('click', () => {
    imageUpload.click();
});

imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const name = personName.value.trim();
    if (!name) {
        setupStatus.textContent = "Please enter a name first.";
        imageUpload.value = '';
        return;
    }

    const reader = new FileReader();
    reader.onload = async (event) => {
        const imageB64 = event.target.result;
        
        // Show preview
        imagePreview.style.backgroundImage = `url(${imageB64})`;
        imagePreview.style.display = 'block';
        setupStatus.textContent = "Uploading image...";

        try {
            const res = await fetch(`${API_URL}/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, image: imageB64 })
            });
            const data = await res.json();
            
            if (data.status === 'success') {
                setupStatus.textContent = `Image uploaded and saved for ${name}!`;
                setTimeout(() => {
                    imagePreview.style.display = 'none';
                }, 2000);
            } else {
                setupStatus.textContent = `Error: ${data.message}.`;
                imagePreview.style.display = 'none';
            }
        } catch (err) {
            setupStatus.textContent = "Network error while uploading.";
            imagePreview.style.display = 'none';
        }
        imageUpload.value = '';
    };
    reader.readAsDataURL(file);
});

trainBtn.addEventListener('click', async () => {
    showView('loading');
    stopCamera();
    
    // min 3 seconds
    let progress = 0;
    progressBar.style.width = '0%';
    
    const progressInterval = setInterval(() => {
        progress += 2; 
        if (progress > 95) progress = 95;
        progressBar.style.width = `${progress}%`;
    }, 50);

    const startTime = Date.now();
    
    try {
        const res = await fetch(`${API_URL}/train`, { method: 'POST' });
        const data = await res.json();
        
        const elapsedTime = Date.now() - startTime;
        if (elapsedTime < 3000) {
            await new Promise(r => setTimeout(r, 3000 - elapsedTime));
        }
        
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        
        if (data.status === 'success') {
            setTimeout(() => {
                startDemo();
            }, 500);
        } else {
            alert(`Training failed: ${data.message}`);
            showView('setup');
            startCamera(setupVideo);
        }
        
    } catch (e) {
        clearInterval(progressInterval);
        alert("Training failed. Check console.");
        showView('setup');
        startCamera(setupVideo);
    }
});

async function startDemo() {
    showView('demo');
    await startCamera(demoVideo);
    
    lastPerson.textContent = 'None';
    updateLock('Unknown');

    if (recognitionInterval) clearInterval(recognitionInterval);
    recognitionInterval = setInterval(processRecognition, 1000); // 1FPS
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
            if (data.name !== 'Unknown') {
                lastPerson.textContent = data.name;
            }
            updateLock(data.name);
        }
    } catch (e) {
        console.error("Recognition error", e);
    }
}

function updateLock(name) {
    const isLocked = name === 'Unknown';
    const h1 = lockStatus.querySelector('h1');
    
    if (isLocked) {
        lockStatus.className = 'lock-status locked';
        h1.textContent = 'LOCKED';
    } else {
        lockStatus.className = 'lock-status unlocked';
        h1.textContent = 'UNLOCKED';
    }
}

async function loadTeamMembers() {
    try {
        const res = await fetch(`${API_URL}/team`);
        const data = await res.json();
        if (data.status === 'success') {
            teamMembersList.innerHTML = '';
            data.team.forEach(member => {
                const li = document.createElement('li');
                li.textContent = member;
                teamMembersList.appendChild(li);
            });
        }
    } catch (e) {
        console.error("Failed to load team");
    }
}

restartBtn.addEventListener('click', () => {
    if (recognitionInterval) clearInterval(recognitionInterval);
    resetSystem();
});

// Avoid executing until DOM is fully loaded or script is at end of body.
// It is at the end of body in index.html, so this is fine.
init();
