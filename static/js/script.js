document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const startBtn = document.getElementById("startBtn");
    const captureBtn = document.getElementById("captureBtn");
    const snapBtn = document.getElementById("snapBtn");
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const resultSection = document.getElementById("result-section");
    const resultImage = document.getElementById("resultImage");
    const accuracySpan = document.getElementById("accuracy");
    const detectionsDiv = document.getElementById("detections");
    const loadingText = document.getElementById("loadingText");
    const timeTaken = document.getElementById("timeTaken");

    let currentImageFile = null;
    let cameraStream = null;

    // ✅ Créer bouton Stop Camera dynamiquement
    const stopBtn = document.createElement("button");
    stopBtn.textContent = "Stop Camera";
    stopBtn.style.display = "none";
    captureBtn.insertAdjacentElement("afterend", stopBtn);

    // ✅ Activer bouton Start quand fichier choisi
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            currentImageFile = fileInput.files[0];
            startBtn.disabled = false;
        }
    });

    // ✅ Envoyer image vers Flask
    async function sendImageToServer(imageBlob) {
        loadingText.style.display = "block";
        timeTaken.style.display = "none";
        const startTime = performance.now();

        const formData = new FormData();
        formData.append("image", imageBlob);

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData
            });
            const data = await response.json();

            if (data.error) {
                alert("Error: " + data.error);
                return;
            }

            // Afficher résultats
            resultSection.style.display = "block";
            resultImage.src = data.annotated_image;
            accuracySpan.textContent = data.accuracy;
            detectionsDiv.innerHTML = "";

            data.detections.forEach(det => {
                const p = document.createElement("p");
                p.textContent = `Class: ${det.class}, Confidence: ${det.confidence}, BBox: ${det.bbox}`;
                detectionsDiv.appendChild(p);
            });

            const endTime = performance.now();
            const duration = ((endTime - startTime) / 1000).toFixed(2);
            timeTaken.textContent = `⏱ Processing time: ${duration} seconds`;
            timeTaken.style.display = "block";

        } catch (err) {
            console.error("Upload failed:", err);
            alert("Failed to process image.");
        } finally {
            loadingText.style.display = "none";
        }
    }

    // ✅ Bouton Start Detection
    startBtn.addEventListener("click", () => {
        if (currentImageFile) {
            sendImageToServer(currentImageFile);
        }
    });

    // ✅ Bouton Camera → ouvrir flux vidéo
    captureBtn.addEventListener("click", () => {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                cameraStream = stream;
                video.style.display = "block";
                snapBtn.style.display = "block";
                stopBtn.style.display = "block";
                video.srcObject = stream;
            })
            .catch(err => {
                console.error("Camera error:", err);
                alert("Cannot access camera.");
            });
    });

    // ✅ Bouton Snap Photo → prendre photo et envoyer
    snapBtn.addEventListener("click", () => {
        const context = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(blob => {
            if (blob) {
                sendImageToServer(blob);
            }
        }, "image/jpeg");
    });

    // ✅ Bouton Stop Camera → couper flux vidéo
    stopBtn.addEventListener("click", () => {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            video.style.display = "none";
            snapBtn.style.display = "none";
            stopBtn.style.display = "none";
        }
    });
});
