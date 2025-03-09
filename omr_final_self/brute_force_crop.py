import cv2
import numpy as np
import os
import random


def find_circles(image):
    """Detect circles (OMR bubbles) using HoughCircles."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                               param1=50, param2=30, minRadius=5, maxRadius=20)
    return circles[0, :] if circles is not None else []


def find_number(image):
    """Detect number presence using contours (simple heuristic)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if 0.3 < aspect_ratio < 1.2 and 10 < w < 40 and 10 < h < 40:  # Heuristic size for numbers
            return True
    return False


def validate_question_row(row):
    """Check if a row contains 1 number and 4 circles."""
    circles = find_circles(row)
    number_present = find_number(row)

    return number_present and len(circles) in [4, 5]  # Allowing slight errors in detection


def crop_and_find_best(image, debug_dir):
    """Perform probability-enhanced brute-force cropping and find the best one."""
    height, width, _ = image.shape
    best_crop = None
    max_valid_questions = 0

    # Ensure debug directory exists
    os.makedirs(debug_dir, exist_ok=True)

    for attempt in range(1000):  # 10 brute-force attempts
        if attempt == 0:
            # Initial crop (default values)
            crop_x_start = int(width * 0.28)
            crop_y_start = int(height * 0.32)
            crop_x_end = width - int(width * 0.03)
            crop_y_end = height - int(height * 0.07)
        else:
            # Shift crop slightly based on previous best
            shift_x = random.randint(-10, 10)
            shift_y = random.randint(-10, 10)
            crop_x_start = max(0, crop_x_start + shift_x)
            crop_y_start = max(0, crop_y_start + shift_y)
            crop_x_end = min(width, crop_x_end + shift_x)
            crop_y_end = min(height, crop_y_end + shift_y)

        # Ensure valid cropping dimensions
        if crop_x_start >= crop_x_end or crop_y_start >= crop_y_end:
            print(f"Attempt {attempt + 1}: Invalid crop dimensions! Skipping...")
            continue

        cropped_image = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        # Check if cropped image is empty
        if cropped_image is None or cropped_image.size == 0:
            print(f"Attempt {attempt + 1}: Cropped image is empty, skipping.")
            continue

        cropped_height, cropped_width, _ = cropped_image.shape
        section_width = cropped_width // 4
        total_valid_questions = 0

        for i in range(4):
            x_start = i * section_width
            x_end = (i + 1) * section_width if i < 3 else cropped_width
            section = cropped_image[:, x_start:x_end]
            section_height = section.shape[0] // 40

            for j in range(40):
                y_start = j * section_height
                y_end = (j + 1) * section_height if j < 39 else section.shape[0]
                row = section[y_start:y_end, :]

                if validate_question_row(row):
                    total_valid_questions += 1

        # Save the crop for debugging
        cropped_path = f"{debug_dir}/crop_attempt_{attempt + 1}.png"
        cv2.imwrite(cropped_path, cropped_image)
        print(f"Attempt {attempt + 1}: Saved crop at {cropped_path} with {total_valid_questions} valid questions.")

        # Select the best crop
        if total_valid_questions > max_valid_questions:
            max_valid_questions = total_valid_questions
            best_crop = cropped_image

    return best_crop


# Load the image
image_path = "./resize_based_on_the_similarity/output_question.jpg"
image = cv2.imread(image_path)

# Ensure image is loaded
if image is None:
    print("Error: Image not found or unable to load!")
    exit(1)

# Debug directory
debug_dir = "./debug"
os.makedirs(debug_dir, exist_ok=True)

# Run the improved brute-force cropping
best_cropped_image = crop_and_find_best(image, debug_dir)

# Save the best crop
if best_cropped_image is not None:
    best_crop_path = f"{debug_dir}/best_cropped_omr.png"
    cv2.imwrite(best_crop_path, best_cropped_image)
    print(f"Saved best cropped OMR at: {best_crop_path}")
else:
    print("No valid crop found.")
