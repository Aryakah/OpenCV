#
# import cv2
# import numpy as np
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
#
# MARGIN = 10  # pixels
# ROW_SIZE = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# TEXT_COLOR = (255, 0, 0)  # red
#
#
# def visualize(
#     image,
#     detection_result
# ) -> np.ndarray:
#   """Draws bounding boxes on the input image and return it.
#   Args:
#     image: The input RGB image.
#     detection_result: The list of all "Detection" entities to be visualize.
#   Returns:
#     Image with bounding boxes.
#   """
#   for detection in detection_result.detections:
#     # Draw bounding_box
#     bbox = detection.bounding_box
#     start_point = bbox.origin_x, bbox.origin_y
#     end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
#     cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
#
#     # Draw label and score
#     category = detection.categories[0]
#     category_name = category.category_name
#     probability = round(category.score, 2)
#     result_text = category_name + ' (' + str(probability) + ')'
#     text_location = (MARGIN + bbox.origin_x,
#                      MARGIN + ROW_SIZE + bbox.origin_y)
#     cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
#                 FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
#
#   return image
#
#
# import cv2
#
# IMAGE_FILE = 'Model/pet-3157961_640.jpg'
#
# # Load the image
# #img = cv2.imread(IMAGE_FILE)
#
# # Check if the image was loaded successfully
# # if img is not None:
# #     # Display the image
# #     cv2.imshow("Image", img)
# #     # Wait for a key press and close the window
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
# # else:
# #     print("Error: Could not load the image.")
#
#
# base_options = python.BaseOptions(model_asset_path='Model/efficientdet_lite2.tflite')
# options = vision.ObjectDetectorOptions(base_options=base_options,
#                                        score_threshold=0.5)
# detector = vision.ObjectDetector.create_from_options(options)
#
# image = mp.Image.create_from_file(IMAGE_FILE)
#
# detection_result = detector.detect(image)
# image_copy = np.copy(image.numpy_view())
# annotated_image = visualize(image_copy, detection_result)
# rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
# cv2.imshow("Image", rgb_annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#


import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image


base_options = python.BaseOptions(model_asset_path='Model/efficientdet_lite2.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Display the frame in a window
        # frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame2)

        detection_result = detector.detect(image)
        image_copy = np.copy(image.numpy_view())
        annotated_image = visualize(image_copy, detection_result)
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Image", rgb_annotated_image)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture when done
    cap.release()
    cv2.destroyAllWindows()









