import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import DBSCAN


def load_and_preprocess(image_path):
    """Enhanced image loading and preprocessing"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or invalid format")

    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned


def detect_braille_dots(binary_image):
    """Improved dot detection with size and circularity filters"""
    contours, _ = cv2.findContours(binary_image,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    dot_centers = []
    dot_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if 20 < area < 200 and 0.7 < circularity < 1.3:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dot_centers.append((cx, cy))
                dot_areas.append(area)

    return np.array(dot_centers), np.median(dot_areas)


def cluster_braille_cells(dot_centers, median_dot_size):
    """Advanced clustering using DBSCAN"""
    # Convert to numpy array for processing
    X = np.array(dot_centers)

    # Estimate cell spacing
    if len(X) > 1:
        distances = np.sqrt(np.sum(np.diff(X, axis=0) ** 2, axis=1))
        cell_spacing = np.median(distances)
    else:
        cell_spacing = median_dot_size * 3  # Fallback estimate

    # DBSCAN clustering
    eps = cell_spacing * 1.5
    db = DBSCAN(eps=eps, min_samples=1).fit(X)
    labels = db.labels_

    # Group dots by cell
    cells = defaultdict(list)
    for i, label in enumerate(labels):
        cells[label].append(X[i])

    return cells, cell_spacing


def analyze_braille_cells(cells, cell_spacing):
    """Comprehensive Braille cell analysis"""
    braille_dict = {
        (1, 0, 0, 0, 0, 0): 'a', (1, 1, 0, 0, 0, 0): 'b', (1, 0, 0, 1, 0, 0): 'c',
        (1, 0, 0, 1, 1, 0): 'd', (1, 0, 0, 0, 1, 0): 'e', (1, 1, 0, 1, 0, 0): 'f',
        (1, 1, 0, 1, 1, 0): 'g', (1, 1, 0, 0, 1, 0): 'h', (0, 1, 0, 1, 0, 0): 'i',
        (0, 1, 0, 1, 1, 0): 'j', (1, 0, 1, 0, 0, 0): 'k', (1, 1, 1, 0, 0, 0): 'l',
        (1, 0, 1, 1, 0, 0): 'm', (1, 0, 1, 1, 1, 0): 'n', (1, 0, 1, 0, 1, 0): 'o',
        (1, 1, 1, 1, 0, 0): 'p', (1, 1, 1, 1, 1, 0): 'q', (1, 1, 1, 0, 1, 0): 'r',
        (0, 1, 1, 1, 0, 0): 's', (0, 1, 1, 1, 1, 0): 't', (1, 0, 1, 0, 0, 1): 'u',
        (1, 1, 1, 0, 0, 1): 'v', (0, 1, 0, 1, 1, 1): 'w', (1, 0, 1, 1, 0, 1): 'x',
        (1, 0, 1, 1, 1, 1): 'y', (1, 0, 1, 0, 1, 1): 'z'
    }

    contraction_symbols = {'and': '⠯', 'for': '⠿', 'of': '⠷',
                           'the': '⠮', 'with': '⠾', 'ch': '⠡'}

    results = {
        'text': '',
        'cells': [],
        'grade': 1,
        'contractions': [],
        'cell_stats': []
    }

    for cell_id, dots in cells.items():
        # Calculate cell bounding box
        dots_array = np.array(dots)
        min_x, min_y = np.min(dots_array, axis=0)
        max_x, max_y = np.max(dots_array, axis=0)
        cell_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)

        # Determine dot positions (Braille 2x3 grid)
        dot_pattern = [0] * 6
        for (x, y) in dots:
            # Position relative to cell center
            rel_x = x - cell_center[0]
            rel_y = y - cell_center[1]

            # Determine dot position in 2x3 grid
            if rel_x < 0:  # Left column
                if rel_y < -cell_spacing / 3:
                    pos = 0  # Dot 1
                elif rel_y < cell_spacing / 3:
                    pos = 1  # Dot 2
                else:
                    pos = 2  # Dot 3
            else:  # Right column
                if rel_y < -cell_spacing / 3:
                    pos = 3  # Dot 4
                elif rel_y < cell_spacing / 3:
                    pos = 4  # Dot 5
                else:
                    pos = 5  # Dot 6
            dot_pattern[pos] = 1

        # Recognize character
        char = braille_dict.get(tuple(dot_pattern), '?')
        results['text'] += char

        # Store cell information
        cell_info = {
            'center': cell_center,
            'dots': dots,
            'pattern': dot_pattern,
            'character': char
        }
        results['cells'].append(cell_info)
        results['cell_stats'].append({
            'dot_count': len(dots),
            'spacing_variation': np.std([y for (x, y) in dots]),
            'horizontal_spread': max_x - min_x
        })

    # Grade detection
    total_dots = sum(len(cell['dots']) for cell in results['cells'])
    avg_dots_per_cell = total_dots / len(results['cells']) if results['cells'] else 0

    if any(c['character'] in contraction_symbols.values() for c in results['cells']):
        results['grade'] = 2
        results['contractions'] = [c['character'] for c in results['cells']
                                   if c['character'] in contraction_symbols.values()]
    elif avg_dots_per_cell > 3.5:
        results['grade'] = 3
    else:
        results['grade'] = 1

    return results


def visualize_results(image, analysis_results):
    """Enhanced visualization with detailed annotations"""
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw all detected dots
    for cell in analysis_results['cells']:
        for (x, y) in cell['dots']:
            cv2.circle(output, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Draw cell bounding box
        dots_array = np.array(cell['dots'])
        min_x, min_y = np.min(dots_array, axis=0)
        max_x, max_y = np.max(dots_array, axis=0)
        cv2.rectangle(output,
                      (int(min_x - 5), int(min_y - 5)),
                      (int(max_x + 5), int(max_y + 5)),
                      (255, 0, 0), 1)

        # Label each cell with its character
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        cv2.putText(output, cell['character'],
                    (int(center_x), int(center_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Add analysis summary
    grade_text = f"Grade: {analysis_results['grade']}"
    if analysis_results['grade'] > 1:
        grade_text += f" (Contractions: {', '.join(analysis_results['contractions'])})"

    cv2.putText(output, grade_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(output, f"Text: {analysis_results['text']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Create detailed analysis plot
    plt.figure(figsize=(15, 8))

    # Original image with annotations
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Braille Detection Results")
    plt.axis('off')

    # Analysis metrics
    plt.subplot(1, 2, 2)
    if analysis_results['cell_stats']:
        dot_counts = [s['dot_count'] for s in analysis_results['cell_stats']]
        spacing_vars = [s['spacing_variation'] for s in analysis_results['cell_stats']]

        plt.bar(range(len(dot_counts)), dot_counts)
        plt.plot(spacing_vars, 'r-', label='Vertical Spacing Variation')
        plt.title("Cell Analysis Metrics")
        plt.xlabel("Cell Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    return output


def process_braille_image(image_path):
    """Complete processing pipeline"""
    try:
        # Step 1: Image preprocessing
        binary = load_and_preprocess(image_path)

        # Step 2: Braille dot detection
        dot_centers, median_dot_size = detect_braille_dots(binary)
        if len(dot_centers) < 6:
            raise ValueError("Insufficient Braille dots detected")

        # Step 3: Cell clustering
        cells, cell_spacing = cluster_braille_cells(dot_centers, median_dot_size)

        # Step 4: Braille analysis
        analysis = analyze_braille_cells(cells, cell_spacing)

        # Step 5: Visualization
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        visualize_results(original_image, analysis)

        return analysis

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return None


# Example usage
if __name__ == "__main__":
    analysis_results = process_braille_image("braille_braille.png")
    if analysis_results:
        print("\n=== Braille Analysis Report ===")
        print(f"Detected Text: {analysis_results['text']}")
        print(f"Braille Grade: {analysis_results['grade']}")
        if analysis_results['contractions']:
            print(f"Contractions Found: {', '.join(analysis_results['contractions'])}")
        print(f"Total Cells Analyzed: {len(analysis_results['cells'])}")
        print(
            f"Average Dots per Cell: {sum(len(c['dots']) for c in analysis_results['cells']) / len(analysis_results['cells']):.2f}")