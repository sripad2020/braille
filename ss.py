import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

# Braille patterns (same as in image generation)
braille_patterns = {
    'a': [(0, 0)], 'b': [(0, 0), (0, 1)], 'c': [(0, 0), (1, 0)],
    'd': [(0, 0), (1, 0), (1, 1)], 'e': [(0, 0), (1, 1)], 'f': [(0, 0), (0, 1), (1, 0)],
    'g': [(0, 0), (0, 1), (1, 0), (1, 1)], 'h': [(0, 0), (0, 1), (1, 1)], 'i': [(0, 1), (1, 0)],
    'j': [(0, 1), (1, 0), (1, 1)], 'k': [(0, 0), (2, 0)], 'l': [(0, 0), (0, 1), (2, 0)],
    'm': [(0, 0), (1, 0), (2, 0)], 'n': [(0, 0), (1, 0), (2, 0), (1, 1)], 'o': [(0, 0), (1, 1), (2, 0)],
    'p': [(0, 0), (0, 1), (1, 0), (2, 0)], 'q': [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    'r': [(0, 0), (0, 1), (1, 1), (2, 0)], 's': [(0, 1), (1, 0), (2, 0)], 't': [(0, 1), (1, 0), (1, 1), (2, 0)],
    'u': [(0, 0), (2, 0), (2, 1)], 'v': [(0, 0), (0, 1), (2, 0), (2, 1)],
    'w': [(0, 1), (1, 1), (2, 0), (2, 1)], 'x': [(0, 0), (1, 0), (2, 0), (2, 1)],
    'y': [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)], 'z': [(0, 0), (1, 1), (2, 0), (2, 1)],
    ' ': []
}


# Step 1: Generate Synthetic Training Data
def generate_synthetic_braille_data(num_samples_per_char=100):
    cell_width, cell_height = 30, 50
    X_data = []
    y_data = []
    for char, pattern in braille_patterns.items():
        for _ in range(num_samples_per_char):
            img = np.ones((cell_height, cell_width), dtype=np.uint8) * 255
            for (row, col) in pattern:
                x = 5 + col * 10
                y = 5 + row * 10
                cv2.circle(img, (x, y), 5, 0, -1)
            img = cv2.resize(img, (28, 28))
            X_data.append(img)
            y_data.append(char)

    X_data = np.array(X_data, dtype=np.float32).reshape(-1, 28, 28, 1) / 255.0
    y_data = np.array(y_data)
    print(f"Generated {len(X_data)} samples with shape {X_data.shape}, labels shape {y_data.shape}")
    print(f"X_data min: {X_data.min()}, max: {X_data.max()}")
    return X_data, y_data


# Step 2: Create and Train Braille Classification Model
def create_and_train_braille_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(27, activation='softmax')  # 26 letters + space
    ])

    X, y = generate_synthetic_braille_data()
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"y_encoded shape: {y_encoded.shape}, unique values: {np.unique(y_encoded)}")

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_encoded, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    return model, le


# Step 3: Detect Braille Dots
def detect_braille_dots(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return np.array([]), (0, 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if 8 < w < 14 and 8 < h < 14:
            dots.append([x + w // 2, y + h // 2])

    print(f"Detected {len(dots)} dots in {image_path}")
    return np.array(dots), img.shape


# Step 4: Process Dots into Cells
def process_to_cells(dots, img_shape, cell_width=30, cell_height=50):
    if len(dots) == 0:
        return []

    cells = []
    dots = dots[dots[:, 0].argsort()]
    for x in range(10, img_shape[1] - 10, cell_width):
        cell_dots = dots[(dots[:, 0] >= x) & (dots[:, 0] < x + cell_width) &
                         (dots[:, 1] >= 10) & (dots[:, 1] < 10 + cell_height)]
        cells.append((x, cell_dots))
        print(f"Cell at x={x}: {len(cell_dots)} dots")

    return cells


# Step 5: Predict Braille Characters
def predict_braille(model, le, cell_img):
    if cell_img.size == 0 or cell_img.shape != (50, 30):
        return ' '
    cell_img = cv2.resize(cell_img, (28, 28)).astype(np.float32) / 255.0
    cell_img = np.expand_dims(cell_img, axis=(0, -1))
    pred = model.predict(cell_img, verbose=0)
    return le.inverse_transform([np.argmax(pred)])[0]


# Step 6: Main Pipeline
def braille_to_text(image_path, model, le):
    dots, img_shape = detect_braille_dots(image_path)
    if len(dots) == 0 and img_shape == (0, 0):
        return "No Braille dots detected or invalid image"

    cells = process_to_cells(dots, img_shape)
    img = cv2.imread(image_path, 0)
    if img is None:
        return "Failed to load image"

    output_text = []
    for x, _ in cells:
        cell_img = img[10:10 + 50, x:x + 30]
        if cell_img.shape != (50, 30):
            output_text.append(' ')
            continue
        char = predict_braille(model, le, cell_img)
        output_text.append(char)

    result = ''.join(output_text)
    print(f"Final extracted text from {image_path}: {result}")
    return result


# Step 7: Generate Braille Image
def generate_braille_image(text, output_path="braille.png"):
    cell_width, cell_height = 30, 50
    img_width = len(text) * cell_width + 20
    img = np.ones((cell_height + 20, img_width, 3), dtype=np.uint8) * 255
    for i, char in enumerate(text.lower()):
        if char not in braille_patterns:
            continue
        for (row, col) in braille_patterns[char]:
            x = 10 + i * cell_width + col * 10
            y = 10 + row * 10
            cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
    cv2.imwrite(output_path, img)
    print(f"Generated: {output_path}")


# Main execution
if __name__ == "__main__":
    # Generate test images
    generate_braille_image("abcde", "braille_abcde.png")
    generate_braille_image("hello world", "braille_hello.png")
    generate_braille_image("test", "braille_test.png")

    # Train model
    model, le = create_and_train_braille_model()

    # Test pipeline
    for image_path in ["braille_abcde.png", "braille_hello.png", "braille_test.png"]:
        result = braille_to_text(image_path, model, le)
        print(f"Extracted Text from {image_path}: {result}")