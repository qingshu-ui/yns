document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('imageFile');
    const uploadButton = document.getElementById('uploadButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorMessage = document.getElementById('errorMessage');
    const responseContainer = document.getElementById('responseContainer');
    const responseData = document.getElementById('responseData');

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        if (!fileInput.files[0]) {
            showError('Please select an image file');
            return;
        }

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);

        uploadButton.disabled = true;
        loadingIndicator.classList.remove('hidden');
        errorMessage.classList.add('hidden');
        responseContainer.classList.add('hidden');

        fetch('/text-select.captcha/reason', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Server responded with an error');
                }
                return response.json();
            })
            .then(data => {
                responseData.textContent = JSON.stringify(data, null, 2);
                responseContainer.classList.remove('hidden');
            })
            .catch(error => {
                showError('An error occurred while uploading the image');
                console.error('Error:', error);
            })
            .finally(() => {
                uploadButton.disabled = false;
                loadingIndicator.classList.add('hidden');
            });
    });

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
    }
});