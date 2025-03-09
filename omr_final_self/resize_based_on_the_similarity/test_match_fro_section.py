import cv2
import numpy as np

def crop_section(omr_image_path, section_image_path):
    # Load the OMR sheet and the section to match
    omr_image = cv2.imread(omr_image_path)
    section_image = cv2.imread(section_image_path, 0)  # Load in grayscale

    # Convert OMR image to grayscale
    omr_gray = cv2.cvtColor(omr_image, cv2.COLOR_BGR2GRAY)

    # Template Matching
    result = cv2.matchTemplate(omr_gray, section_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Get coordinates of the best match
    start_x, start_y = max_loc
    h, w = section_image.shape

    # Crop that section from the OMR image
    cropped_section = omr_image[start_y:start_y + h, start_x:start_x + w]

    return cropped_section

# Example Usage
cropped = crop_section("output.jpg", "../input/cropped_omr_v2.png")
cv2.imwrite("cropped_output.jpg", cropped)
