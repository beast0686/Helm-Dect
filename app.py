from flask import Flask, request, render_template, redirect, url_for, session, jsonify, send_from_directory
import mysql.connector
from mysql.connector import errorcode
import os
import cv2
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Database configuration
db_config = {
    'user': 'root',
    'password': 'root123',
    'host': 'localhost',
    'database': 'helmet_detection',
}


# Function to connect to the database
def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        if conn.is_connected():
            print('Connected to MySQL database')
            return conn
        else:
            print('Connection failed')
            return None
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        return None


# Define the folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Paths to the YOLO model files (update these paths as needed)
weights0_path = r'D:\BNMIT\Semester 4\Python\Project\Helmet detection\datasets\detect-person-on-motorbike-or-scooter\yolov3-obj_final.weights'
configuration0_path = r'D:\BNMIT\Semester 4\Python\Project\Helmet detection\datasets\detect-person-on-motorbike-or-scooter\yolov3_pb.cfg'
weights1_path = r'D:\BNMIT\Semester 4\Python\Project\Helmet detection\datasets\helmet-detection-yolov3\yolov3-helmet.weights'
configuration1_path = r'D:\BNMIT\Semester 4\Python\Project\Helmet detection\datasets\helmet-detection-yolov3\yolov3-helmet.cfg'
labels0_path = r'D:\BNMIT\Semester 4\Python\Project\Helmet detection\datasets\detect-person-on-motorbike-or-scooter\coco.names'
labels1_path = r'D:\BNMIT\Semester 4\Python\Project\Helmet detection\datasets\helmet-detection-yolov3\helmet.names'

# Load the YOLO models
network0 = cv2.dnn.readNetFromDarknet(configuration0_path, weights0_path)
network1 = cv2.dnn.readNetFromDarknet(configuration1_path, weights1_path)

# Load the class labels
labels0 = open(labels0_path).read().strip().split('\n')
labels1 = open(labels1_path).read().strip().split('\n')

# Get the output layer names
layers_names0_output = [network0.getLayerNames()[i - 1] for i in network0.getUnconnectedOutLayers()]
layers_names1_output = [network1.getLayerNames()[i - 1] for i in network1.getUnconnectedOutLayers()]

# Define constants for detection
probability_minimum = 0.5  # Minimum probability to filter weak detections
threshold = 0.3  # Threshold for non-maxima suppression


# Home route
@app.route('/')
def index():
    # Redirect to login if user is not logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            return redirect(url_for('index'))
        else:
            return render_template('login_error.html')
    return render_template('login.html')


# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Hash the password using the default method (pbkdf2:sha256)
        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)', (name, email, hashed_password))
        conn.commit()
        cursor.close()
        conn.close()

        return redirect(url_for('login'))

    return render_template('signup.html')


# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))


# Contact route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Insert feedback into the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO feedback (name, email, message) VALUES (%s, %s, %s)',
            (name, email, message)
        )
        conn.commit()
        cursor.close()
        conn.close()

        return redirect(url_for('contact'))

    return render_template('contact.html')


# About route
@app.route('/about')
def about():
    # Redirect to login if user is not logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('about.html')


# Upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    # Redirect to login if user is not logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is one of the allowed types/extensions
    if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load the uploaded image
        image_input = cv2.imread(file_path)
        h, w = image_input.shape[:2]
        blob = cv2.dnn.blobFromImage(image_input, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Set the input for the YOLO networks
        network0.setInput(blob)
        network1.setInput(blob)

        # Run the forward pass to get the detections
        output_from_network0 = network0.forward(layers_names0_output)
        output_from_network1 = network1.forward(layers_names1_output)

        np.random.seed(42)  # Set the random seed for reproducibility
        colours0 = np.random.randint(0, 255, size=(len(labels0), 3), dtype='uint8')
        colours1 = np.random.randint(0, 255, size=(len(labels1), 3), dtype='uint8')

        bounding_boxes0 = []  # List to hold bounding boxes for person detection
        confidences0 = []  # List to hold confidences for person detection
        class_numbers0 = []  # List to hold class numbers for person detection

        bounding_boxes1 = []  # List to hold bounding boxes for helmet detection
        confidences1 = []  # List to hold confidences for helmet detection
        class_numbers1 = []  # List to hold class numbers for helmet detection

        # Process detection results for person detection
        for result in output_from_network0:
            for detection in result:
                scores = detection[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]
                if confidence_current > probability_minimum:
                    box_current = detection[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current.astype('int')
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))
                    bounding_boxes0.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences0.append(float(confidence_current))
                    class_numbers0.append(class_current)

        # Process detection results for helmet detection
        for result in output_from_network1:
            for detection in result:
                scores = detection[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]
                if confidence_current > probability_minimum:
                    box_current = detection[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current.astype('int')
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))
                    bounding_boxes1.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences1.append(float(confidence_current))
                    class_numbers1.append(class_current)

        # Non-Maximum Suppression for person detection
        results0 = cv2.dnn.NMSBoxes(bounding_boxes0, confidences0, probability_minimum, threshold)
        if len(results0) > 0:
            for i in results0.flatten():
                x_min, y_min = bounding_boxes0[i][0], bounding_boxes0[i][1]
                box_width, box_height = bounding_boxes0[i][2], bounding_boxes0[i][3]
                colour_box_current = [int(j) for j in colours0[class_numbers0[i]]]
                cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current,
                              5)
                text_box_current0 = '{}: {:.4f}'.format(labels0[int(class_numbers0[i])], confidences0[i])
                cv2.putText(image_input, text_box_current0, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            colour_box_current, 5)

        # Non-Maximum Suppression for helmet detection
        results1 = cv2.dnn.NMSBoxes(bounding_boxes1, confidences1, probability_minimum, threshold)
        if len(results1) > 0:
            for i in results1.flatten():
                x_min, y_min = bounding_boxes1[i][0], bounding_boxes1[i][1]
                box_width, box_height = bounding_boxes1[i][2], bounding_boxes1[i][3]
                colour_box_current = [int(j) for j in colours1[class_numbers1[i]]]
                cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current,
                              5)
                text_box_current1 = '{}: {:.4f}'.format(labels1[int(class_numbers1[i])], confidences1[i])
                cv2.putText(image_input, text_box_current1, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            colour_box_current, 5)

        # Save the output image with detections
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + file.filename)
        cv2.imwrite(output_path, image_input)

        return jsonify({'message': 'Detection complete', 'image_url': 'uploads/output_' + file.filename})
    else:
        return jsonify({'error': 'Invalid file type'})


# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
