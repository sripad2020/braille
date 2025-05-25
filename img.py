import cv2
import numpy as np
from collections import defaultdict

# ====================== BRAILLE MAPPINGS ======================
english_to_braille = {
    'a': '100000', 'b': '110000', 'c': '100100', 'd': '100110', 'e': '100010',
    'f': '110100', 'g': '110110', 'h': '110010', 'i': '010100', 'j': '010110',
    'k': '101000', 'l': '111000', 'm': '101100', 'n': '101110', 'o': '101010',
    'p': '111100', 'q': '111110', 'r': '111010', 's': '011100', 't': '011110',
    'u': '101001', 'v': '111001', 'w': '010111', 'x': '101101', 'y': '101111',
    'z': '101011', ' ': '000000'
}

# Create reverse mapping
braille_to_english = {v: k for k, v in english_to_braille.items()}


# ====================== BRAILLE GENERATION ======================
def generate_braille_image(text, output_path="braille_output.png", dot_radius=5, cell_width=20, cell_height=30,
                           padding=10):
    """Generate Braille image from English text"""
    text = text.lower()
    braille_patterns = [english_to_braille.get(char, '000000') for char in text if char in english_to_braille]

    img_width = len(braille_patterns) * (cell_width + padding) + padding
    img_height = cell_height + 2 * padding
    img = np.ones((img_height, img_width), dtype=np.uint8) * 255

    for i, pattern in enumerate(braille_patterns):
        cell_x = padding + i * (cell_width + padding)
        cell_y = padding

        positions = [
            (cell_x + cell_width // 4, cell_y + cell_height // 6),  # 1
            (cell_x + cell_width // 4, cell_y + cell_height // 2),  # 2
            (cell_x + cell_width // 4, cell_y + 5 * cell_height // 6),  # 3
            (cell_x + 3 * cell_width // 4, cell_y + cell_height // 6),  # 4
            (cell_x + 3 * cell_width // 4, cell_y + cell_height // 2),  # 5
            (cell_x + 3 * cell_width // 4, cell_y + 5 * cell_height // 6)  # 6
        ]

        for j, dot in enumerate(pattern):
            if dot == '1':
                cv2.circle(img, (int(positions[j][0]), int(positions[j][1])), dot_radius, 0, -1)

    cv2.imwrite(output_path, img)
    return output_path


# ====================== BRAILLE DETECTION ======================
def preprocess_image(image_path):
    """Prepare image for Braille detection"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or invalid path")

    # Enhanced preprocessing
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def extract_braille_cells(image_path):
    """Detect and group Braille dots into cells"""
    processed = preprocess_image(image_path)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get dot centers
    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 100:
            x, y, w, h = cv2.boundingRect(cnt)
            dots.append((x + w // 2, y + h // 2))

    if not dots:
        return []

    # Sort dots by y then x
    dots_sorted = sorted(dots, key=lambda x: (x[1], x[0]))

    # Group into lines
    lines = []
    current_line = []
    y_threshold = 15

    for dot in dots_sorted:
        if not current_line or dot[1] - current_line[-1][1] < y_threshold:
            current_line.append(dot)
        else:
            lines.append(sorted(current_line, key=lambda x: x[0]))
            current_line = [dot]
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x[0]))

    # Group into cells (2x3 grids)
    cells = []
    for line in lines:
        current_cell = []
        x_threshold = 20

        for dot in line:
            if not current_cell or dot[0] - current_cell[-1][0] < x_threshold:
                current_cell.append(dot)
            else:
                cells.append(current_cell)
                current_cell = [dot]
        if current_cell:
            cells.append(current_cell)

    return cells


def translate_braille(cells):
    """Convert detected Braille cells to English text"""
    text = ""
    for cell in cells:
        pattern = ['0'] * 6
        if not cell:
            text += ' '
            continue

        # Get cell boundaries
        min_x = min(dot[0] for dot in cell)
        min_y = min(dot[1] for dot in cell)
        max_y = max(dot[1] for dot in cell)
        cell_height = max(1, max_y - min_y)

        # Map dots to positions
        for dot in cell:
            x, y = dot
            col = 0 if (x - min_x) < cell_height / 2 else 1
            row = min(2, int(3 * (y - min_y) / cell_height))
            pos = row * 2 + col
            if pos < 6:
                pattern[pos] = '1'

        # Look up pattern
        text += braille_to_english.get(''.join(pattern), '?')

    return text


# ====================== MAIN PROGRAM ======================
if __name__ == "__main__":
    # 1. Generate Braille image
    text = "hello world"
    braille_image = generate_braille_image(text)
    print(f"Generated Braille image for '{text}': {braille_image}")

    # 2. Translate Braille back to English
    cells = extract_braille_cells(braille_image)
    if cells:
        translated = translate_braille(cells)
        print(f"Translated text: {translated}")
    else:
        print("No Braille cells detected")