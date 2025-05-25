from flask import Flask, render_template, request, redirect
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Grade 1 Braille mapping (6-dot binary pattern to letter)
# pattern is a tuple of 6 bits: dots 1 to 6 in order
braille_to_char = {
    (1,0,0,0,0,0): 'a',
    (1,1,0,0,0,0): 'b',
    (1,0,0,1,0,0): 'c',
    (1,0,0,1,1,0): 'd',
    (1,0,0,0,1,0): 'e',
    (1,1,0,1,0,0): 'f',
    (1,1,0,1,1,0): 'g',
    (1,1,0,0,1,0): 'h',
    (0,1,0,1,0,0): 'i',
    (0,1,0,1,1,0): 'j',
    (1,0,1,0,0,0): 'k',
    (1,1,1,0,0,0): 'l',
    (1,0,1,1,0,0): 'm',
    (1,0,1,1,1,0): 'n',
    (1,0,1,0,1,0): 'o',
    (1,1,1,1,0,0): 'p',
    (1,1,1,1,1,0): 'q',
    (1,1,1,0,1,0): 'r',
    (0,1,1,1,0,0): 's',
    (0,1,1,1,1,0): 't',
    (1,0,1,0,0,1): 'u',
    (1,1,1,0,0,1): 'v',
    (0,1,0,1,1,1): 'w',
    (1,0,1,1,0,1): 'x',
    (1,0,1,1,1,1): 'y',
    (1,0,1,0,1,1): 'z',
}

# Neighbors relative positions to find adjacent dots in grid
def get_neighbors(dot, dots, max_dist=20):
    neighbors = []
    x, y = dot
    for nx, ny in dots:
        if (nx, ny) != (x, y):
            dist = np.sqrt((nx - x) ** 2 + (ny - y) ** 2)
            if dist < max_dist:
                neighbors.append((nx, ny))
    return neighbors


def connected_components(dots, max_dist=20):
    """
    Groups dots into connected clusters (Braille cells) based on neighbor distance.
    Returns list of groups, each group is list of dot coordinates.
    """
    unvisited = set(dots)
    groups = []

    while unvisited:
        stack = [unvisited.pop()]
        group = []

        while stack:
            current = stack.pop()
            group.append(current)
            neighbors = get_neighbors(current, unvisited, max_dist)
            for n in neighbors:
                if n in unvisited:
                    unvisited.remove(n)
                    stack.append(n)
        groups.append(group)
    return groups


def parse_braille_pattern(cell):
    """
    Given a list of 6 dot coords for a Braille cell, determine which dots are raised.
    The dot positions correspond to:

    Dot positions in a 2x3 grid (x=col, y=row):

    1 4
    2 5
    3 6

    We assign dots by sorting coords top-to-bottom, left-to-right.

    Return tuple of 6 bits (1 if raised, else 0).
    """
    if len(cell) != 6:
        return None  # incomplete cell

    # Sort dots by Y (row) then X (col)
    sorted_cell = sorted(cell, key=lambda p: (p[1], p[0]))

    # Identify columns by median X split
    xs = [p[0] for p in sorted_cell]
    median_x = np.median(xs)

    pattern = [0]*6  # dots 1-6

    for i, (x, y) in enumerate(sorted_cell):
        # Determine dot number by row and col
        row = i // 2  # 0,1,2 rows
        col = 0 if x < median_x else 1
        dot_number = row * 2 + col  # 0-based

        if dot_number < 6:
            pattern[dot_number] = 1

    return tuple(pattern)


@app.route('/', methods=['GET', 'POST'])
def index():
    decoded_text = ''
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load image in grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Threshold to binary image (black dots on white background)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Find connected components (dots)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

        # Filter out small areas - keep only blobs of a size typical for Braille dots
        dots = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 10 < area < 500:
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                dots.append((cx, cy))

        # Group dots into Braille cells
        groups = connected_components(dots, max_dist=30)

        # Decode each group
        decoded_chars = []
        for cell in groups:
            if len(cell) == 6:
                pattern = parse_braille_pattern(cell)
                if pattern and pattern in braille_to_char:
                    decoded_chars.append(braille_to_char[pattern])
                else:
                    decoded_chars.append('?')
            else:
                decoded_chars.append('?')

        decoded_text = ''.join(decoded_chars)

        os.remove(filepath)

    return render_template('home.html', decoded_text=decoded_text)


if __name__ == '__main__':
    app.run()
