# Helmet Detection System

## Overview

This Helmet Detection System is a web application built with Flask that allows users to upload images, which are then processed to detect persons on motorbikes or scooters and whether they are wearing helmets. The application uses YOLO (You Only Look Once) models for object detection and stores user information and feedback in a MySQL database.

## Features

- User authentication (login and signup)
- Image upload and processing
- Detection of persons on motorbikes or scooters
- Detection of helmets on detected persons
- User feedback submission
- Responsive design

## Technologies Used

- Flask (Python web framework)
- MySQL (database)
- OpenCV (computer vision library)
- YOLO (object detection models)
- HTML/CSS/JavaScript (frontend)
- Bootstrap (CSS framework for responsive design)

## Prerequisites

- Python 3.x
- MySQL
- OpenCV
- Flask and related Python libraries (`flask`, `mysql-connector-python`, `werkzeug`)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/helmet-detection-system.git
    cd helmet-detection-system
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the MySQL database:
    - Create a database named `helmet_detection`.
    - Create the following tables:
        ```sql
        CREATE TABLE users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            email VARCHAR(255) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL
        );

        CREATE TABLE feedback (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            email VARCHAR(255) NOT NULL,
            message TEXT NOT NULL
        );
        ```

4. Update the database configuration in the code: Modify the `db_config` dictionary in the `app.py` file with your MySQL credentials:
    ```python
    db_config = {
        'user': 'your-username',
        'password': 'your-password',
        'host': 'localhost',
        'database': 'helmet_detection',
    }
    ```

5. Set up the YOLO model files:
    - Download the YOLO model weights, configuration files, and class labels.
    - Update the paths to these files in the `app.py` file:
        ```python
        weights0_path = 'path/to/yolov3-obj_final.weights'
        configuration0_path = 'path/to/yolov3_pb.cfg'
        weights1_path = 'path/to/yolov3-helmet.weights'
        configuration1_path = 'path/to/yolov3-helmet.cfg'
        labels0_path = 'path/to/coco.names'
        labels1_path = 'path/to/helmet.names'
        ```

6. Create the `uploads` directory:
    ```bash
    mkdir uploads
    ```

## Usage

1. Run the Flask application:
    ```bash
    python app.py
    ```

2. Access the application: Open your web browser and go to [http://192.168.107.240:8000](http://192.168.107.240:8000).

3. Sign up and log in:
    - Create a new account by signing up.
    - Log in with your credentials.

4. Upload an image:
    - Go to the upload page and select an image file (PNG, JPG, JPEG).
    - The application will process the image and display the results with detected persons and helmets.

5. Submit feedback:
    - Go to the contact page and submit your feedback.

## File Structure

- `app.py`: Main application file containing the Flask routes and functionality.
- `templates/`: Directory containing HTML templates.
    - `index.html`: Home page template.
    - `login.html`: Login page template.
    - `signup.html`: Signup page template.
    - `contact.html`: Contact page template.
    - `about.html`: About page template.
    - `login_error.html`: Login error page template.
- `static/`: Directory for static files (CSS, JavaScript, images).
- `uploads/`: Directory for storing uploaded and processed images.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- YOLO (You Only Look Once) model for object detection.
- OpenCV library for image processing.
- Flask web framework for building the web application.
- Bootstrap for responsive design.
