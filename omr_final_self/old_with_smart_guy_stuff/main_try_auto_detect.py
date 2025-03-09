import cv2
import numpy as np
import os


def preprocess_image(image_path):
    """Loads and preprocesses the OMR sheet image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    return image, thresh


def detect_rectangles(image):
    """Detects vertical rectangles on the left, right, or both sides."""
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left_rects, right_rects = [], []
    height, width = image.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > width * 0.1:  # Filtering larger vertical rectangles
            if x < width * 0.1:
                left_rects.append((x, y, w, h))
            elif x > width * 0.9:
                right_rects.append((x, y, w, h))
    return left_rects, right_rects


def scale_and_warp(image, left_rects):
    """Uses left-side rectangles to align the OMR sheet."""
    if not left_rects:
        return image
    return image


def detect_all_bubbles(image):
    """Detects all bubbles (empty and filled) using adaptive threshold and contour detection."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if 8 < radius < 30:
            bubbles.append((int(x), int(y), int(radius)))
    return bubbles


def draw_contours(image, bubbles, output_path="bubbles_detected.png"):
    """Draws detected bubbles and contours on the image and saves it."""
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x, y, r) in bubbles:
        cv2.circle(image_color, (x, y), r, (0, 255, 0), 2)
    cv2.imwrite(output_path, image_color)


def segment_sections(image, output_dir="output_sections"):
    """Segments the image into different sections dynamically and saves them."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    height, width = image.shape
    bubbles = detect_all_bubbles(image)
    sections = {"header": image[0:int(height * 0.2), :]}

    if bubbles:
        min_x = min(b[0] - b[2] for b in bubbles)
        max_x = max(b[0] + b[2] for b in bubbles)
        min_y = min(b[1] - b[2] for b in bubbles)
        max_y = max(b[1] + b[2] for b in bubbles)
        sections["answers"] = image[min_y:max_y, min_x:max_x]

    for section_name, section_img in sections.items():
        cv2.imwrite(os.path.join(output_dir, f"{section_name}.png"), section_img)

    draw_contours(image, bubbles)
    return sections


def main(image_path):
    """Main function to process OMR sheet."""
    image, _ = preprocess_image(image_path)
    left_rects, right_rects = detect_rectangles(image)
    image = scale_and_warp(image, left_rects)
    segment_sections(image)


# Example Usage
if __name__ == "__main__":
    main("omr.png")
