import collections
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def fix_row_alignment(rows, median_radius, expected_cols=4):
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
                fixed_row.append(np.array([int(expected_x), int(row_median_y), median_radius], dtype=np.uint16))

        fixed_rows.append(sorted(fixed_row, key=lambda c: c[0]))  # Ensure sorted order

    return fixed_rows


def add_missing_circles(image, cluster_map, median_radius, debug_output_path="./debug/04_corrected_circles.png"):
    # Detect and cluster circles
    image = image.copy()

    # Process each section
    section_rows = []
    fix_circle = set()
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
        fixed_rows = fix_row_alignment(rows, median_radius)

        tuple_fixed_rows = set([(x, y, r) for row in fixed_rows for x, y, r in row])
        tuple_rows = set([(x, y, r) for row in rows for x, y, r in row])

        for circle in tuple_fixed_rows:
            if not circle in tuple_rows:
                fix_circle.add(circle)

        section_rows.append(fixed_rows)

    row_number = 1
    # Step 5: Draw the corrected circles
    for section in section_rows:
        for row in section:

            for circle in row:
                x, y, r = circle
                if (x, y, r) in fix_circle:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)

                x, y, r = int(x), int(y), int(r)
                cv2.circle(image, (x, y), r, color, 2)  # Blue outline

            if row:
                x_text, y_text, _ = map(int, row[0])  # Get first circle position
                cv2.putText(image, str(row_number), (x_text - 30, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                row_number += 1  # Increment row counter

    # Save the final corrected image
    cv2.imwrite(debug_output_path, image)
    print(f"Corrected image saved as '{debug_output_path}'")

    return section_rows
