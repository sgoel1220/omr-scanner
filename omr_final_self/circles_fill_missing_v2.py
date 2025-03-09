import collections
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Load the image
image_path = "question_bubbles.jpg"
# image_path = "./final_code/debug/crop_to_questions.png"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)


def transform_to_option_arr(section):
    ans_arr = []
    for _ in range(len(section[0][0])):
        ans_arr.append([])

    for section in section:
        for options in section:
            for i, option in enumerate(options):
                ans_arr[i].append(option)

    return ans_arr



def cluster_circles(image):
    """ Detect and cluster circles into 4 vertical sections. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=50,
        param2=22,
        minRadius=5,
        maxRadius=10
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle_centers = circles[0, :, :2]  # Extract (x, y) coordinates

        # Calculate median radius
        median_radius = int(np.median(circles[0, :, 2]))  # Ensure integer

        # Use only X-coordinates for clustering
        x_values = circle_centers[:, 0].reshape(-1, 1)

        # Apply K-Means clustering with 4 vertical groups
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(x_values)
        cluster_centers = kmeans.cluster_centers_

        # Calculate distances of each point to its cluster center
        distances = cdist(x_values, cluster_centers)
        min_distances = np.min(distances, axis=1)

        # Use IQR to filter outliers
        Q1, Q3 = np.percentile(min_distances, [25, 75])
        IQR = Q3 - Q1
        threshold = Q3 + 0.6 * IQR

        # Keep only inliers
        filtered_indices = np.where(min_distances <= threshold)[0]
        filtered_circles = circles[0][filtered_indices]
        filtered_labels = labels[filtered_indices]

        label_map = collections.defaultdict(list)

        # Draw circles
        for i, (x, y, r) in enumerate(filtered_circles):
            label_map[filtered_labels[i]].append([x, y, median_radius])  # Use median radius
            cv2.circle(image, (x, y), median_radius, (255, 0, 0), 2)  # Blue outline
            cv2.circle(image, (x, y), 2, (0, 255, 0), 3)  # Green center point

        cv2.imwrite("clustered_circles.png", image)
        print("Processed image saved as 'clustered_circles.png'")

        return label_map, median_radius

    return {}, 10  # Default median radius if no circles are found


# def fix_row_alignment(rows, expected_cols=4):
#     """ Identify missing or misaligned circles and correct them. """
#
#     fixed_rows = []
#
#     # Step 1: Flatten all rows to analyze column positions
#     all_circles = [circle for row in rows for circle in row]
#
#     # Step 2: Group circles by approximate X-coordinates (columns)
#     x_positions = sorted([c[0] for c in all_circles])
#     column_medians = np.median(np.array(x_positions).reshape(-1, expected_cols), axis=0)
#
#     for row in rows:
#         if len(row) == expected_cols:
#             fixed_rows.append(row)  # Row is already complete
#             continue
#
#         # Step 3: Sort circles by x-coordinate
#         row.sort(key=lambda c: c[0])
#         known_x = np.array([c[0] for c in row])
#         known_y = np.array([c[1] for c in row])
#
#         # Step 4: Check each expected position
#         fixed_row = []
#         used_indices = set()
#
#         for i, expected_x in enumerate(column_medians):
#             # Find closest existing circle
#             closest_index = np.argmin(np.abs(known_x - expected_x)) if len(known_x) > 0 else -1
#             if closest_index not in used_indices and abs(known_x[closest_index] - expected_x) < 15:
#                 fixed_row.append(row[closest_index])  # Keep the aligned circle
#                 used_indices.add(closest_index)
#             else:
#                 # Approximate missing circle
#                 row_median_y = np.median(known_y) if len(known_y) > 0 else 0  # Use median Y of the row
#                 fixed_row.append([int(expected_x), int(row_median_y), median_radius])
#
#         fixed_rows.append(sorted(fixed_row, key=lambda c: c[0]))  # Ensure sorted order
#
#     return fixed_rows


def fix_row_alignment(rows, expected_cols=4):
    """ Identify missing or misaligned circles and correct them. """

    fixed_rows = []

    # Step 1: Extract all X-coordinates to determine column positions
    all_circles = [circle for row in rows for circle in row]
    x_positions = sorted(set(c[0] for c in all_circles))  # Unique sorted X values

    # Step 2: Use K-Means clustering to group columns dynamically
    if len(x_positions) >= expected_cols:
        kmeans = KMeans(n_clusters=expected_cols, random_state=42, n_init=10)
        labels = kmeans.fit_predict(np.array(x_positions).reshape(-1, 1))
        column_medians = [np.median([x_positions[i] for i in range(len(x_positions)) if labels[i] == cluster])
                          for cluster in range(expected_cols)]
    else:
        # If fewer detected columns than expected, use available X-values
        column_medians = x_positions

    column_medians.sort()  # Ensure columns are in left-to-right order

    for row in rows:
        if len(row) == expected_cols:
            fixed_rows.append(row)  # Row is already complete
            continue

        # Step 3: Sort circles by x-coordinate
        row.sort(key=lambda c: c[0])
        known_x = np.array([c[0] for c in row])
        known_y = np.array([c[1] for c in row])

        # Step 4: Check each expected position
        fixed_row = []
        used_indices = set()

        for expected_x in column_medians:
            # Find closest existing circle
            closest_index = np.argmin(np.abs(known_x - expected_x)) if len(known_x) > 0 else -1
            if closest_index not in used_indices and abs(known_x[closest_index] - expected_x) < 15:
                fixed_row.append(row[closest_index])  # Keep the aligned circle
                used_indices.add(closest_index)
            else:
                # Approximate missing circle using median Y of the row
                row_median_y = np.median(known_y) if len(known_y) > 0 else 0
                fixed_row.append([int(expected_x), int(row_median_y), median_radius])

        fixed_rows.append(sorted(fixed_row, key=lambda c: c[0]))  # Ensure sorted order

    return fixed_rows



import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import linregress

def fix_row_alignment_with_tillt(rows, expected_cols=4):
    """
    Identify missing or misaligned circles and correct them,
    while accounting for tilt in both x and y directions,
    preventing overlaps and ensuring all missing circles are added.
    """

    fixed_rows = []

    # Step 1: Extract all X, Y coordinates
    all_circles = [circle for row in rows for circle in row]
    x_positions = np.array([c[0] for c in all_circles])
    y_positions = np.array([c[1] for c in all_circles])

    # Step 2: Estimate column positions using K-Means clustering
    if len(set(x_positions)) >= expected_cols:
        kmeans = KMeans(n_clusters=expected_cols, random_state=42, n_init=10)
        labels = kmeans.fit_predict(x_positions.reshape(-1, 1))
        column_medians = [np.median(x_positions[labels == cluster]) for cluster in range(expected_cols)]
    else:
        column_medians = sorted(set(x_positions))  # Use unique x-values if not enough clusters

    column_medians.sort()  # Ensure left-to-right order

    # Step 3: Estimate row positions using linear regression (tilt correction)
    slope, intercept, _, _, _ = linregress(y_positions, x_positions)  # Fit a line to get tilt
    print(f"Estimated X-Axis Tilt: Slope={slope}, Intercept={intercept}")

    def correct_x(y_val):
        """ Adjust x-position based on estimated tilt. """
        return slope * y_val + intercept

    for row in rows:
        if len(row) == expected_cols:
            fixed_rows.append(row)  # Row is already correct
            continue

        # Step 4: Sort row by x-coordinate
        row.sort(key=lambda c: c[0])
        known_x = np.array([c[0] for c in row])
        known_y = np.array([c[1] for c in row])

        # Step 5: Correct expected X positions based on tilt
        expected_x_positions = [correct_x(np.median(known_y)) + (i * median_radius * 2) for i in range(expected_cols)]

        # Step 6: Identify missing circles and add them
        fixed_row = []
        used_indices = set()

        for expected_x in expected_x_positions:
            closest_index = np.argmin(np.abs(known_x - expected_x)) if len(known_x) > 0 else -1
            if closest_index not in used_indices and abs(known_x[closest_index] - expected_x) < median_radius * 1.5:
                fixed_row.append(row[closest_index])  # Keep correct circle
                used_indices.add(closest_index)
            else:
                # Approximate missing circle using row's median Y value
                row_median_y = int(np.median(known_y)) if len(known_y) > 0 else 0

                # Ensure no overlap by checking minimum distance
                if not any(abs(c[0] - expected_x) < median_radius * 1.5 for c in fixed_row):
                    fixed_row.append([int(expected_x), row_median_y, median_radius])

        fixed_rows.append(sorted(fixed_row, key=lambda c: c[0]))  # Ensure left-to-right order

    return fixed_rows


# Load image
image = cv2.imread(image_path, cv2.IMREAD_COLOR)


def add_missing_circles(image, cluster_map):
    # Detect and cluster circles


    # Process each section
    section_rows = []
    # Step 1: Sort clusters based on the leftmost circle in each cluster
    sorted_clusters = sorted(cluster_map.values(), key=lambda circles: min(c[0] for c in circles))

    for circles in sorted_clusters:
        # Step 1: Sort circles by y-coordinate
        sorted_circles = sorted(circles, key=lambda c: c[1])

        # Step 2: Group into rows based on median y-values
        rows = []
        current_row = [sorted_circles[0]] if sorted_circles else []

        for circle in sorted_circles[1:]:
            current_median_y = np.median([c[1] for c in current_row])
            if abs(current_median_y - circle[1]) < 10:  # Adjust threshold if needed
                current_row.append(circle)
            else:
                rows.append(current_row)
                current_row = [circle]

        if current_row:
            rows.append(current_row)

        # Step 3: Sort each row by x-coordinate
        for row in rows:
            row.sort(key=lambda c: c[0])

        # Step 4: Fix misaligned or missing circles
        fixed_rows = fix_row_alignment(rows)

        section_rows.append(fixed_rows)

    # Step 5: Draw the corrected circles
    # Step 5: Draw the corrected circles and number the rows
    row_number = 1  # Initialize row counter

    for section in section_rows:
        for row in section:
            for x, y, r in row:
                x, y, r = int(x), int(y), int(r)
                cv2.circle(image, (x, y), r, (255, 0, 0), 2)  # Blue outline

            # Place the row number near the first circle of the row
            if row:
                x_text, y_text, _ = map(int, row[0])  # Get first circle position
                cv2.putText(image, str(row_number), (x_text - 30, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                row_number += 1  # Increment row counter

    # Save the final corrected image
    cv2.imwrite("corrected_circles.png", image)
    print("Corrected image saved as 'corrected_circles.png'")

    return section_rows

def detect_marked_circles(circles, image, current_running_option="a",
                          standard_deviation=2, output_path="marked_image.jpg"):
    deviations = []
    intensities = []

    # Convert image to grayscale if not already
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Extract median intensity for each circle
    for (x, y, r) in circles:
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)

        # Get pixel intensities inside the circle
        pixels = gray[mask == 255]
        median_intensity = np.median(pixels)
        intensities.append(median_intensity)

    # Calculate the overall median intensity and standard deviation
    overall_median = np.median(intensities)
    std_dev = np.std(intensities)
    threshold = standard_deviation * std_dev  # Using 2 standard deviations

    # Identify marked circles
    marked_circles = []
    question_ans = {}
    for i, (x, y, r) in enumerate(circles):
        deviation = abs(intensities[i] - overall_median)
        deviations.append(deviation)

        if deviation > threshold:  # Threshold based on standard deviation
            question_ans[i] = current_running_option
            marked_circles.append((x, y, r))
            cv2.circle(image, (int(x), int(y)), int(r), (255, 0, 0), 2)  # Draw in black

    print("marked", marked_circles)
    print("marked", len(marked_circles))
    # Save output image
    cv2.imwrite(output_path, image)

    return marked_circles, deviations, question_ans


cluster_map, median_radius = cluster_circles(image.copy())
final_sections = add_missing_circles(image.copy(), cluster_map)


circles = transform_to_option_arr(final_sections)
ans = ["" for _ in range(162)]

for i in range(4):
    # TODO edge case of 2 circles marked, pick the most deviation one
    _, _, cur_ans = detect_marked_circles(circles[i], image, current_running_option=chr(ord("a") + i),
                                    standard_deviation=1.5)

    for key in cur_ans:
        ans[key + 1] = cur_ans[key]


print(ans)

