import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tensorflow.keras.models import load_model
import string
import tensorflow as tf

# Define the ClueboardDetector class
class ClueboardDetector:
    def __init__(self):
        self.bridge = CvBridge()
        
        self.interpreter = tf.lite.Interpreter(model_path='quantized_model.tflite')

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback)
        self.score_publisher = rospy.Publisher('/score_tracker', String, queue_size=1)

        self.characters = " " + string.ascii_uppercase + string.digits
        self.label_dict = {char: i for i, char in enumerate(self.characters)}
        self.index_to_char = {index: char for char, index in self.label_dict.items()}

    def detect_clueboard(self, cv_image):
        # Convert to HSV color space for easier color thresholding
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 125, 40])  # Lower bound for dark blue
        upper_blue = np.array([140, 255, 255]) # Upper bound for dark blue

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the second outermost contour
        if len(contours)>0
            # Sort the contours by area in descending order (largest first)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            if len(sorted_contours)>1:
                needed_contour = sorted_contours[1]
                epsilon = 0.05 * cv2.arcLength(needed_contour, True)
                approx = np.squeeze(cv2.approxPolyDP(needed_contour, epsilon, True))
                approx = order_points(approx)

                if len(approx) == 4:  # Ensure the contour has 4 points
                    # Assuming the detected contour approximates the corners of the signboard
                    pts1 = np.float32([approx[0], approx[1], approx[2], approx[3]])
                    # Define points for the desired output (signboard dimensions)
                    signboard_width, signboard_height = 600, 400 
                    pts2 = np.float32([[0, 0], [signboard_width, 0], [signboard_width, signboard_height], [0, signboard_height]])
                    # Calculate the perspective transform matrix and apply it
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    signboard_transformed = cv2.warpPerspective(cv_image, matrix, (signboard_width, signboard_height))
                    return signboard_transformed
        return None  # Return None if no signboard is detected

    def shutdown_hook(self):
        rospy.loginfo("ClueboardDetector shutdown.")

    # Define a function to preprocess and extract letters from the detected signboard
    def preprocess_and_extract_letters(self, signboard_image):

        image = cv2.cvtColor(signboard_image, cv2.COLOR_BGR2GRAY)
        image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        signboard_image = cv2.addWeighted(image, 1.5, laplacian, -0.5, 0)
        
        if len(signboard_image.shape) == 3:
            h, w, _ = signboard_image.shape  # For color images
        else:
            h, w = signboard_image.shape  # For grayscale images

        category_top_crop = int(0.05 * h)  # Calculate the number of rows to crop from the top
        category_bottom_crop = int(0.65 * h)  # Calculate the number of rows to crop from the bottom

        category_right_crop = int(0.13 * w)  # Calculate the number of columns to crop from the right
        category_left_crop = int(0.42 * w)  # Calculate the number of columns to crop from the left

        cropped_image_category = signboard_image[category_top_crop:h - category_bottom_crop, category_left_crop:w - category_right_crop]

        word_top_crop = int(0.55 * h)  # Calculate the number of rows to crop from the top
        word_bottom_crop = int(0.15 * h)  # Calculate the number of rows to crop from the bottom

        word_right_crop = int(0.05 * w)  # Calculate the number of columns to crop from the right
        word_left_crop = int(0.05 * w)  # Calculate the number of columns to crop from the left

        cropped_image_word = signboard_image[word_top_crop:h - word_bottom_crop, word_left_crop:w - word_right_crop]

        _, w1 = cropped_image_category.shape
        _, w2 = cropped_image_word.shape

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
            resized_img = cv2.resize(letter_img, (45, 120))
            resized_img = np.expand_dims(resized_img, axis=0)  # Add batch dimension
            resized_img = np.float32(resized_img)  # Convert to FLOAT32 and normalize
            preprocessed_letters_category.append(resized_img)

        # Preprocess letters for 'word'
        for letter_img in letters_word:
            resized_img = cv2.resize(letter_img, (45, 120))
            resized_img = np.expand_dims(resized_img, axis=0)  # Add batch dimension
            resized_img = np.float32(resized_img)  # Convert to FLOAT32 and normalize
            preprocessed_letters_word.append(resized_img)

        # Convert lists to numpy arrays
        preprocessed_letters_category = np.array(preprocessed_letters_category)
        preprocessed_letters_word = np.array(preprocessed_letters_word)

        return preprocessed_letters_category, preprocessed_letters_word

    def run_tflite_model(self, input_data):
        input_data = np.expand_dims(input_data, axis=-1)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data


# Callback function for the image subscriber
def callback(ros_image):
    try:
        # Convert the ROS image to an OpenCV image
        cv_image = detect.bridge.imgmsg_to_cv2(ros_image, desired_encoding = "bgr8")
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    signboard_image = detect.detect_clueboard(cv_image)

    if signboard_image is not None:
        #Preprocess the signboard image and extract letters
        preprocessed_letters_category, preprocessed_letters_word = detect.preprocess_and_extract_letters(signboard_image)

        # Use the CNN model to predict letters
        predicted_labels_category = []
        predicted_labels_word = []

        for letter in preprocessed_letters_category:
            letter = detect.run_tflite_model(letter)
            predicted_labels_category.append(letter)

        for letter in preprocessed_letters_word:
            letter = detect.run_tflite_model(letter)
            predicted_labels_word.append(letter)

        # Assuming predicted_labels_category and predicted_labels_word contain model predictions
        predicted_indices_category = [np.argmax(letter) for letter in predicted_labels_category]
        predicted_indices_word = [np.argmax(letter) for letter in predicted_labels_word]

        # Decode the predictions to characters
        decoded_category = ''.join(detect.index_to_char[index] for index in predicted_indices_category)

        decoded_word = ''.join(detect.index_to_char[index] for index in predicted_indices_word)
        
        categories = ["SIZE", "VICTIM", "CRIME", "TIME", "PLACE", "MOTIVE", "WEAPON", "BANDIT"]

        # Initialize a variable to store the found index
        location = None

        # Use a for loop with enumerate to find the index
        for index, string in enumerate(categories):
            if string[0] == decoded_category[0]:
                if string[1] == decoded_category[1]:
                    location = index+1
                    break  # Exit the loop once the string is found

        # Construct the message with team information and prediction to send to the score tracker
        team_id = "1W3B"
        team_password = "pword"
        clue_location = location
        clue_prediction = decoded_word
        message_data = f"{team_id},{team_password},{clue_location},{clue_prediction}"
        message = String(data=message_data)
        detect.score_publisher.publish(message)

def order_points(pts):
    if len(pts.shape) >= 2 and pts.shape[1] == 2:

        #Calculate the sum and difference of the points' coordinates
        sum_pts = pts.sum(axis=1)
        diff_pts = np.diff(pts, axis=1)

        #The bottom-left point will have the smallest sum, whereas
        # the top-right point will have the largest sum
        top_left = pts[np.argmin(sum_pts)]
        bottom_right = pts[np.argmax(sum_pts)]

        # The top-left point will have the smallest difference,
        # whereas the bottom-right point will have the largest difference
        top_right = pts[np.argmin(diff_pts)]
        bottom_left = pts[np.argmax(diff_pts)]

        # Return the coordinates in the order: top-left, top-right, bottom-right, bottom-left
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

if __name__ == '__main__':

    # Initialize the ROS node
    rospy.init_node('clue_detection', anonymous=True)
    detect = ClueboardDetector()

    rospy.on_shutdown(detect.shutdown_hook)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down ClueboardDetector node.")
        cv2.destroyAllWindows()