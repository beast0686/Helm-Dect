document.addEventListener('DOMContentLoaded', function () {
    const darkModeToggle = document.getElementById('dark-mode-toggle');

    // Add event listener for toggle change
    darkModeToggle.addEventListener('change', function () {
        if (darkModeToggle.checked) {
            enableDarkMode();
        } else {
            disableDarkMode();
        }
    });

    // Check local storage for dark mode preference on page load
    if (localStorage.getItem('darkMode') === 'enabled') {
        darkModeToggle.checked = true;
        enableDarkMode();
    }

    // Function to enable dark mode
    function enableDarkMode() {
        document.body.classList.add('dark-mode');
        document.querySelectorAll('.container').forEach(function (container) {
            container.classList.add('dark-mode');
        });
        localStorage.setItem('darkMode', 'enabled');
    }

    // Function to disable dark mode
    function disableDarkMode() {
        document.body.classList.remove('dark-mode');
        document.querySelectorAll('.container').forEach(function (container) {
            container.classList.remove('dark-mode');
        });
        localStorage.setItem('darkMode', null);
    }
});
