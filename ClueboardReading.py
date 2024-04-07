import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tensorflow.keras.models import load_model

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
            # Draw bounding rectangle around the outer contour
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
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

    return cropped_image_category, cropped_image_word


# Initialize the ROS node
rospy.init_node('clue_detection')

# Initialize the CvBridge
bridge = CvBridge()

# Initialize the publisher for /score_tracker topic
score_publisher = rospy.Publisher('/score_tracker', String, queue_size=10)

# Load the pre-trained CNN model
cnn_model = load_model('path_to_your_saved_model.h5')

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
        cropped_image_category, cropped_image_word = preprocess_and_extract_letters(signboard_image)

        # Use the CNN model to predict letters
        # Here you would predict with your CNN model and format the output into a string clue_prediction

        predicted_labels_category = cnn_model(cropped_image_category)

        predicted_labels_word = cnn_model(cropped_image_word)

        category = ""
        word = ""

        count_blanks_category = 0
        count_blanks_word = 0
        i = 0
        j = 0

        for i in range(6):
            
            if count_blanks_category == 2:
                break
            
            category += predicted_labels_category[i]

            if predicted_labels_category[i] == " ":
                count_blanks_category++

        for j in range(12):

            if count_blanks_word == 2:
                break
            
            word += predicted_labels_word[i]

            if predicted_labels_word[i] == " ":
                count_blanks_word++
        
        categories = ["SIZE", "VICTIM", "CRIME", "TIME", "PLACE", "MOTIVE", "WEAPON", "BANDIT"]

        # Initialize a variable to store the found index
        location = None

        # Use a for loop with enumerate to find the index
        for index, string in enumerate(categories):
            if string == category:
                location = index+1
                break  # Exit the loop once the string is found

        # Construct the message with team information and prediction to send to the score tracker
        team_id = "1White3Brown"       # Replace with your actual team ID
        team_password = "YourPass"   # Replace with your actual team password
        clue_location = location           # Replace with the actual location of the clue
        #clue_prediction = "PREDICTED_CLUE"  # Replace with the actual predicted clue from the model
        message_data = f"{team_id},{team_password},{clue_location},{word}"
        message = String(data=message_data)
        score_publisher.publish(message)

# Subscribe to the image topic provided by the robot's camera
image_subscriber = rospy.Subscriber('/robot_camera/image_raw', Image, image_callback)

# Keep the script running
rospy.spin()
