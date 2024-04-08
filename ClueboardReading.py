import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tensorflow.keras.models import load_model
import string

# Define the ClueboardDetector class
class ClueboardDetector:
    def __init__(self):
        pass

    def detect_clueboard(self, cv_image):
        # Convert image to grayscale
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # Perform Canny edge detection
        edges = cv2.Canny(blurred_image, 50, 150)
        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the outermost contour
        if contours:
            # Sort contours by area in descending order
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            outer_contour = contours[0]  # The first contour will be the largest
            x, y, w, h = cv2.boundingRect(outer_contour)
            # Extract the signboard region from the image
            signboard = cv_image[y:y+h, x:x+w]
            return signboard
        return None  # Return None if no signboard is detected

# Define a function to preprocess and extract letters from the detected signboard
def preprocess_and_extract_letters(signboard_image):
    # Assume the image is evenly split into 4 parts (Adapted from your provided split_image function)
    # ...

    h, w = signboard_image.shape

    category_top_crop = int(0.05 * h)  # Calculate the number of rows to crop from the top
    category_bottom_crop = int(0.65 * h)  # Calculate the number of rows to crop from the bottom

    category_right_crop = int(0.13 * w)  # Calculate the number of columns to crop from the right
    category_left_crop = int(0.42 * w)  # Calculate the number of columns to crop from the left

    cropped_image_category = signboard_image[category_top_crop:h - category_bottom_crop, category_left_crop:w - category_right_crop]

    word_top_crop = int(0.55 * h)  # Calculate the number of rows to crop from the top
    word_bottom_crop = int(0.15 * h)  # Calculate the number of rows to crop from the bottom

    word_right_crop = int(0.05 * w)  # Calculate the number of columns to crop from the right
    word_left_crop = int(0.05*w)  # Calculate the number of columns to crop from the left

    cropped_image_word = signboard_image[word_top_crop:h - word_bottom_crop, word_left_crop:w - word_right_crop]

    h1, w1 = cropped_image_category.shape
    h2, w2 = cropped_image_word.shape

    letter_width_category = int(w1 / 6)  # Calculate the width of each letter (category)
    letter_width_word = int(w2 / 12)  # Calculate the width of each letter (word)

    # Splitting the letters
    letters_category = [cropped_image_category[:, i * letter_width_category: (i + 1) * letter_width_category] for i in range(6)]
    letters_word = [cropped_image_word[:, i * letter_width_word: (i + 1) * letter_width_word] for i in range(12)]

    # Initialize lists to store preprocessed letter images
    preprocessed_letters_category = []
    preprocessed_letters_word = []

    # Preprocess letters for 'category'
    for letter_img in letters_category:
        # Resize each letter image to match the model's input shape: (120, 45)
        resized_img = cv2.resize(letter_img, (45, 120))
        # Ensure it's a single channel (grayscale), though it should already be
        if len(resized_img.shape) == 3:
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        
        # Add a channel dimension to match the input shape (120, 45, 1)
        resized_img = np.expand_dims(resized_img, axis=-1)
        # Append to the list
        preprocessed_letters_category.append(resized_img)

    # Preprocess letters for 'word'
    for letter_img in letters_word:
        # Repeat the preprocessing steps as above for 'category'
        resized_img = cv2.resize(letter_img, (45, 120))
        if len(resized_img.shape) == 3:
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        resized_img = np.expand_dims(resized_img, axis=-1)
        preprocessed_letters_word.append(resized_img)

    # Convert lists to numpy arrays
    preprocessed_letters_category = np.array(preprocessed_letters_category)
    preprocessed_letters_word = np.array(preprocessed_letters_word)

    return preprocessed_letters_category, preprocessed_letters_word


# Initialize the ROS node
rospy.init_node('clue_detection')

# Initialize the CvBridge
bridge = CvBridge()

# Initialize the publisher for /score_tracker topic
score_publisher = rospy.Publisher('/score_tracker', String, queue_size=10)

# Load the pre-trained CNN model
cnn_model = load_model('https://github.com/pranaydang/BoardReader/blob/main/cnn_model.keras')

# Assuming label_dict maps characters to indices

characters = " " + string.ascii_uppercase + string.digits
label_dict = {char: i for i, char in enumerate(characters)}

# Create an inverse mapping
index_to_char = {index: char for char, index in label_dict.items()}

# Callback function for the image subscriber
def image_callback(ros_image):
    try:
        # Convert the ROS image to an OpenCV image
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        print(e)
        return

    # Instantiate the ClueboardDetector and detect the signboard in the image
    detector = ClueboardDetector()
    signboard_image = detector.detect_clueboard(cv_image)

    if signboard_image is not None:
        # Preprocess the signboard image and extract letters
        preprocessed_letters_category, preprocessed_letters_word = preprocess_and_extract_letters(signboard_image)

        # Use the CNN model to predict letters
        # Here you would predict with your CNN model and format the output into a string clue_prediction

        predicted_labels_category = cnn_model.predict(preprocessed_letters_category)

        predicted_labels_word = cnn_model.predict(preprocessed_letters_word)

        # Assuming predicted_labels_category and predicted_labels_word contain model predictions
        predicted_indices_category = np.argmax(predicted_labels_category, axis=1)
        predicted_indices_word = np.argmax(predicted_labels_word, axis=1)

        # Decode the predictions to characters
        decoded_category = ''.join(index_to_char[index] for index in predicted_indices_category)
        decoded_word = ''.join(index_to_char[index] for index in predicted_indices_word)
        
        categories = ["SIZE", "VICTIM", "CRIME", "TIME", "PLACE", "MOTIVE", "WEAPON", "BANDIT"]

        # Initialize a variable to store the found index
        location = None

        # Use a for loop with enumerate to find the index
        for index, string in enumerate(categories):
            if string[0] == decoded_category[0]:
                location = index+1
                break  # Exit the loop once the string is found

        # Construct the message with team information and prediction to send to the score tracker
        team_id = "1W3B"       # Replace with your actual team ID
        team_password = "pword"   # Replace with your actual team password
        clue_location = location           # Replace with the actual location of the clue
        clue_prediction = decoded_word # Replace with the actual predicted clue from the model
        message_data = f"{team_id},{team_password},{clue_location},{clue_prediction}"
        message = String(data=message_data)
        score_publisher.publish(message)

# Subscribe to the image topic provided by the robot's camera
image_subscriber = rospy.Subscriber('/robot_camera/image_raw', Image, image_callback)

# Keep the script running
rospy.spin()
