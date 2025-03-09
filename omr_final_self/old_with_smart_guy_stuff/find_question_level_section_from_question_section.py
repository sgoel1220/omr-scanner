

import cv2
import numpy as np
import os
from collections import Counter

from main import preprocess_image_for_questions



def main():
    image_path = "debug/9_answer_section.png"
    quest_section_folder = "./debug/question_section"
    save_path = "debug/question_section/"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    preprocess_image_for_questions(image, quest_section_folder)


    # Load the thresholded OMR sheet image
    image_path = quest_section_folder +  "/4_thresholded.png"  # Update with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get image dimensions
    height, width = image.shape

    # Define vertical split positions
    # Define vertical split positions
    quest_width = width
    section_1 = image[:, width // 3.4:]  # First quarter

    cv2.imwrite(save_path + "onlyq.png", section_1)
    # section_2 = image[:, width // 4: width // 2]  # Second quarter
    # section_3 = image[:, width // 2: 3 * width // 4]  # Third quarter
    # section_4 = image[:, 3 * width // 4:]  # Fourth quarter
    #
    # # Save the sections
    # cv2.imwrite(save_path + "section_1.png", section_1)
    # cv2.imwrite(save_path + "section_2.png", section_2)
    # cv2.imwrite(save_path + "section_3.png", section_3)
    # cv2.imwrite(save_path + "section_4.png", section_4)


main()




# import cv2
# import numpy as np
# import os
#
# # Define paths
# input_dir = "./debug/question_section"
# output_dir = "./debug/question_section"
# image_path = os.path.join(input_dir, "4_thresholded.png")
#
# # Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)
#
# # Load the image
# image = cv2.imread(image_path)
# # cv2.imwrite(os.path.join(output_dir, "1_original.png"), image)
# #
# # # Convert to grayscale
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # cv2.imwrite(os.path.join(output_dir, "2_grayscale.png"), gray)
# #
# # # Apply thresholding to get a binary image
# _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
# # cv2.imwrite(os.path.join(output_dir, "3_thresholded.png"), thresh)
#
# kernel = np.ones((3, 3), np.uint8)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
#
# cv2.imwrite(os.path.join(output_dir, "2_morphologyEx.png"), thresh)
# # Find contours
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Sort contours from top to bottom
# contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
#
# # Draw contours for debugging
# contour_debug = image.copy()
# cv2.drawContours(contour_debug, contours, -1, (0, 255, 0), 2)
# cv2.imwrite(os.path.join(output_dir, "4_contours.png"), contour_debug)
#
# # Extract and save sections
# extracted_sections = []
# for i, ctr in enumerate(contours):
#     x, y, w, h = cv2.boundingRect(ctr)
#
#     # Ignore small contours
#     if h > 50 and w > 50:
#         cropped_section = image[y:y + h, x:x + w]
#         extracted_sections.append(cropped_section)
#         cv2.imwrite(os.path.join(output_dir, f"5_section_{i}.png"), cropped_section)
#
# print(f"Extracted {len(extracted_sections)} sections and saved in {output_dir}.")
