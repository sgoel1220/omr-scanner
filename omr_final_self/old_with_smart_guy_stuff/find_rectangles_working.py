import cv2
import numpy as np
import os
from collections import Counter


def preprocess_image(image_path, debug_folder="debug"):
    """Loads and preprocesses the OMR sheet image."""
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

    print("Loading image...")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(debug_folder, "1_original.png"), image)

    print("Applying CLAHE for contrast enhancement...")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    cv2.imwrite(os.path.join(debug_folder, "2_clahe_applied.png"), image)

    print("Applying Gaussian blur...")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(os.path.join(debug_folder, "3_blurred.png"), blurred)

    print("Applying adaptive threshold...")
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite(os.path.join(debug_folder, "4_thresholded.png"), thresh)

    return image, thresh

# Ensure debug directory exists
debug_dir = "checkpoints"
os.makedirs(debug_dir, exist_ok=True)

# Load the image
image_path = "omr_v2.png"  # Update path if needed

image, thresh = preprocess_image(image_path, debug_dir)

print("Applying threshold for rectangle detection...")
_, thresh = cv2.threshold(thresh, 150, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite(os.path.join(debug_dir, "5_thresholded_for_rectangles.png"), thresh)

print("Applying morphological transformations...")
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imwrite(os.path.join(debug_dir, "6_morphology_applied.png"), thresh)

# Get image dimensions and crop the left third
height, width = thresh.shape
left_third = thresh[:, :width // 3]

# Threshold to get binary image
_, thresh = cv2.threshold(left_third, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get bounding boxes and areas of all detected contours
rectangles = [cv2.boundingRect(cnt) for cnt in contours]  # (x, y, w, h)
areas = [w * h for _, _, w, h in rectangles]

# Sort rectangles by area (biggest first)
sorted_rectangles = sorted(zip(areas, rectangles), key=lambda x: -x[0])

# Find most frequent large rectangle size
area_counts = Counter([round(area, -2) for area, _ in sorted_rectangles])  # Group areas to nearest 100
most_common_large_area = max(area_counts, key=lambda x: (area_counts[x], x))  # Pick highest frequency, largest

# Filter rectangles that match this common large size (±100 pixels)
filtered_rectangles = [rect for area, rect in sorted_rectangles if most_common_large_area - 100 <= area <= most_common_large_area + 100]

# Ensure rectangles are sorted top-to-bottom
filtered_rectangles = sorted(filtered_rectangles, key=lambda rect: rect[1])  # Sort by y-coordinate

# Extract rectangle centers
centers = [(x + w // 2, y + h // 2) for x, y, w, h in filtered_rectangles]

# Find the "best-fit" vertical line
x_positions = [x for x, _ in centers]
median_x = int(np.median(x_positions))  # Find the median x-position as the center reference

# Keep only rectangles close to the vertical line
vertical_tolerance = 10  # Adjust tolerance if needed
aligned_rectangles = [rect for rect in filtered_rectangles if abs((rect[0] + rect[2] // 2) - median_x) <= vertical_tolerance]

# Convert grayscale to BGR for colored output
output_image = cv2.cvtColor(left_third, cv2.COLOR_GRAY2BGR)

# Draw and number selected rectangles
for idx, (x, y, w, h) in enumerate(aligned_rectangles, start=1):
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
    cv2.putText(output_image, str(idx), (x + w + 5, y + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Number the rectangle

# Save the output image
output_path = os.path.join(debug_dir, "aligned_rectangles.png")
cv2.imwrite(output_path, output_image)
print(f"Processed image saved at: {output_path}")
