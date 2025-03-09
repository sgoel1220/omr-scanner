import cv2
import os
import numpy as np
def align_rectangles(image):
    import cv2
    import numpy as np
    import os
    from collections import Counter
    print(image)
    # Ensure it's in grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get image dimensions and crop the left third
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    height, width = image.shape
    left_third = image[:, :width // 3]

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
    filtered_rectangles = [rect for area, rect in sorted_rectangles if
                           most_common_large_area - 100 <= area <= most_common_large_area + 100]
    #
    # # Ensure rectangles are sorted top-to-bottom
    # filtered_rectangles = sorted(filtered_rectangles, key=lambda rect: rect[1])  # Sort by y-coordinate
    #
    # # Extract rectangle centers
    # centers = [(x + w // 2, y + h // 2) for x, y, w, h in filtered_rectangles]
    #
    # # Find the "best-fit" vertical line
    # x_positions = [x for x, _ in centers]
    # median_x = int(np.median(x_positions))  # Find the median x-position as the center reference
    #
    # # Keep only rectangles close to the vertical line
    # vertical_tolerance = 10  # Adjust tolerance if needed
    # aligned_rectangles = [rect for rect in filtered_rectangles if
    #                       abs((rect[0] + rect[2] // 2) - median_x) <= vertical_tolerance]

    # Sort rectangles by their y-coordinate (top-to-bottom)
    filtered_rectangles = sorted(filtered_rectangles, key=lambda rect: rect[1])

    # Extract centers
    centers = [(x + w // 2, y + h // 2) for x, y, w, h in filtered_rectangles]

    # Find the longest nearly vertical sequence using a dynamic approach
    tolerance = 10  # Adjust tolerance for vertical alignment
    longest_sequence = []
    best_sequence = []

    for i in range(len(centers)):
        current_sequence = [filtered_rectangles[i]]
        last_x = centers[i][0]

        for j in range(i + 1, len(centers)):
            x_j, y_j = centers[j]
            if abs(x_j - last_x) <= tolerance:  # Ensure it's roughly vertical
                current_sequence.append(filtered_rectangles[j])
                last_x = x_j  # Update the last x-position

        if len(current_sequence) > len(longest_sequence):
            longest_sequence = current_sequence

    # Final set of rectangles forming the longest nearly straight vertical line
    aligned_rectangles = longest_sequence

    # Convert grayscale to BGR for colored output
    output_image = cv2.cvtColor(left_third, cv2.COLOR_GRAY2BGR)

    # Draw and number selected rectangles
    for idx, (x, y, w, h) in enumerate(aligned_rectangles, start=1):
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
        cv2.putText(output_image, str(idx), (x + w + 5, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Number the rectangle

    # Save the output image
    output_path = os.path.join("./", "aligned_rectangles.png")
    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved at: {output_path}")

    return aligned_rectangles


def preprocess_image(image, debug_folder="debug"):
    """Loads and preprocesses the OMR sheet image."""
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Otsu's thresholding
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, 2)

    cv2.imwrite(os.path.join(debug_folder, "4_thresholded.png"), image)

    return cv2.imread(os.path.join(debug_folder, "4_thresholded.png"))
# image_path = "./cropped_omr.png"

def align_image(image):

    rectangles = align_rectangles(image)

    # print(type(left_markers[0]), left_markers)

    x_coords = [x for x, y, w, h in rectangles]
    y_coords = [y for x, y, w, h in rectangles]

    # Compute the average X position to align to
    avg_x = int(np.mean(x_coords))

    # Define source and destination points for transformation
    src_pts = np.float32([(x, y) for x, y, _, _ in rectangles])
    dst_pts = np.float32([(avg_x, y) for _, y, _, _ in rectangles])

    # Compute the perspective transformation matrix
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # Load the original image (update the path)
    # image = cv2.imread("omr_v2_thresholded.png", cv2.IMREAD_GRAYSCALE)
    rows, cols, _  = image.shape

    # Apply the transformation
    aligned_image = cv2.warpAffine(image, M, (cols, rows))

    # Save and show the aligned image
    cv2.imwrite("omr_v2_aligned.png", aligned_image)
    return aligned_image



def crop_image(image):
    height, width, _ = image.shape

    # Define the cropping region
    crop_x_start = int(width * 0.28)
    crop_y_start = int(height * 0.32)

    crop_x_end = width - int(width * 0.03)
    crop_y_end = height - int(height * 0.07)

    # Crop the bottom-right corner
    cropped_image = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # Save the cropped image
    cropped_image_path = "./debug/cropped_omr.png"
    cv2.imwrite(cropped_image_path, cropped_image)

    # Get dimensions of the cropped image
    cropped_height, cropped_width, _ = cropped_image.shape

    # Divide into 4 vertical sections
    section_width = cropped_width // 4
    total_question_counter = 1
    for i in range(4):
        x_start = i * section_width
        x_end = (i + 1) * section_width if i < 3 else cropped_width  # Ensure last section includes remainder
        section = cropped_image[:, x_start:x_end]

        # Save each vertical section
        section_dir = f"./debug/section_{i + 1}"
        os.makedirs(section_dir, exist_ok=True)
        section_path = f"{section_dir}/section_{i + 1}.png"
        questions_path = "./debug/questions"
        os.makedirs(questions_path, exist_ok=True)
        cv2.imwrite(section_path, section)
        print(f"Saved section {i + 1} at: {section_path}")

        # Divide each vertical section into 40 horizontal parts
        section_height = section.shape[0] // 40

        for j in range(40):
            y_start = j * section_height
            y_end = (j + 1) * section_height if j < 39 else section.shape[0]  # Ensure last row includes remainder
            row = section[y_start:y_end, :]

            # Save each row
            row_path = f"{questions_path}/question_{total_question_counter}.png"
            total_question_counter += 1
            cv2.imwrite(row_path, row)

            print(f"Saved row {j + 1} in section {i + 1} at: {row_path}")

image_path = "./resize_based_on_the_similarity/output_question.jpg"
image = cv2.imread(image_path)

# Ensure the debug directory exists
os.makedirs("./debug", exist_ok=True)

image = preprocess_image(image)
crop_image(image)
