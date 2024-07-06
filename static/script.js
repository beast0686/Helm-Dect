function darkMode() {
    document.body.classList.toggle('dark-mode-active');
}

function detectHelmet() {
    const uploadInput = document.getElementById('uploadInput');

    // Check if a file is selected
    const file = uploadInput.files[0];
    if (!file) {
        alert('Please select an image file.');
        return;
    }

    // Check if the file is an image
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    // Prepare form data for backend
    const formData = new FormData();
    formData.append('file', file);

    // Send the file to the backend for detection
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.message === 'Detection complete') {
            const processedImageUrl = data.image_url;
            window.open(processedImageUrl, '_blank');
        } else {
            alert(data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during the detection process.');
    });
}
