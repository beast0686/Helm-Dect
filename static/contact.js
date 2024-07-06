document.addEventListener('DOMContentLoaded', function () {
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const signup = document.querySelector(".signup");
    const login = document.querySelector(".login");
    const slider = document.querySelector(".btn .slider"); // Ensure the selector targets the slider inside the btn
    const formSection = document.querySelector(".form-section");

    // Dark mode toggle functionality
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
        document.body.classList.add('dark-mode-active');
        localStorage.setItem('darkMode', 'enabled');
    }

    // Function to disable dark mode
    function disableDarkMode() {
        document.body.classList.remove('dark-mode-active');
        localStorage.setItem('darkMode', null);
    }

    // Slider functionality for login/signup toggle
    signup.addEventListener("click", () => {
        slider.classList.add("moveslider");
        formSection.classList.add("form-section-move");
    });

    login.addEventListener("click", () => {
        slider.classList.remove("moveslider");
        formSection.classList.remove("form-section-move");
    });
});
