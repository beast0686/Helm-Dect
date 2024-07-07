---

# Helmet Detection System

This project is a web application built with Flask for detecting persons and helmets in uploaded images using YOLO (You Only Look Once) object detection models. Users can register, login, upload images, and view detection results.

## Features

- **User Authentication**: Users can register, login, and logout securely.
- **Image Upload**: Upload images to detect persons and helmets.
- **Object Detection**: Utilizes YOLO models to detect persons and helmets in uploaded images.
- **Database Integration**: MySQL database is used to store user information and feedback.

## Technologies Used

- **Flask**: Python web framework used for backend development.
- **MySQL**: Database management system for storing user data and feedback.
- **YOLO (You Only Look Once)**: Object detection models for real-time detection of persons and helmets.
- **OpenCV**: Library for image processing and computer vision tasks.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/beast0686/helmet-detection-system.git
   cd helmet-detection-system
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Configure MySQL:

   - Install MySQL and create a database named `helmet_detection`.
   - Update the `db_config` in `app.py` with your MySQL username and password.

4. Set up YOLO models:

   - Update paths to your YOLO model weights, configuration, and label files in `app.py`.

5. Run the application:

   ```
   python app.py
   ```

   Open your web browser and go to `http://localhost:5000` to view the application.

## Usage

- **Register**: Create a new account with your name, email, and password.
- **Login**: Log into your account securely.
- **Upload**: Upload an image file (PNG, JPG, JPEG) to detect persons and helmets.
- **View Results**: View the processed image with bounding boxes and confidence scores.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
