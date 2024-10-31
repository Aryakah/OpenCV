import cv2
import os

# Load the Haar cascade classifier for face detection
cascade_path = 'C:/Users/ichverdienees/Desktop/OpenCV-Dashcam-Car-Detection-master/cascade_dir/cascade.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Check if the cascade classifier has loaded successfully
if face_cascade.empty():
    raise IOError("Error loading cascade classifier. Check the file path.")

# Function to detect and display faces in an image
def detect_faces(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise IOError(f"Error loading image at {image_path}. Please check the file path.")

    # Convert the image to grayscale (Haar cascades require grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    # The detectMultiScale function detects objects in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a blue rectangle with a thickness of 2

    # Display the output image with detected faces
    cv2.imshow('Image with Detected Faces', image)

    # Wait for a key press and then close the displayed image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to execute face detection on multiple images
def main():
    # Specify the directory containing images
    image_directory = 'path_to_your_images_directory'  # Replace with your directory path

    # Iterate over all files in the specified directory
    for filename in os.listdir(image_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image file types
            image_path = os.path.join(image_directory, filename)
            print(f"Processing {image_path}...")
            try:
                detect_faces(image_path)
            except Exception as e:
                print(f"An error occurred while processing {image_path}: {e}")

if __name__ == "__main__":
    main()

