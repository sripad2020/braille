from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from werkzeug.utils import secure_filename
import PyPDF2
import pytesseract
from PIL import Image
import io

app = Flask(__name__)
app.secret_key = "The APP key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    conn.commit()
    conn.close()


init_db()

# Braille translation mappings
braille_to_text = {
    # Letters (a-z)
    '⠁': 'a', '⠃': 'b', '⠉': 'c', '⠙': 'd', '⠑': 'e',
    '⠋': 'f', '⠛': 'g', '⠓': 'h', '⠊': 'i', '⠚': 'j',
    '⠅': 'k', '⠇': 'l', '⠍': 'm', '⠝': 'n', '⠕': 'o',
    '⠏': 'p', '⠟': 'q', '⠗': 'r', '⠎': 's', '⠞': 't',
    '⠥': 'u', '⠧': 'v', '⠺': 'w', '⠭': 'x', '⠽': 'y', '⠵': 'z',

    # Numbers (1-0)
    '⠼⠁': '1', '⠼⠃': '2', '⠼⠉': '3', '⠼⠙': '4', '⠼⠑': '5',
    '⠼⠋': '6', '⠼⠛': '7', '⠼⠓': '8', '⠼⠊': '9', '⠼⠚': '0',

    # Punctuation
    '⠂': ',', '⠲': '.', '⠆': ';', '⠤': '-',
    '⠦': '?', '⠖': '!', '⠶': '"', '⠄': "'",

    # Common contractions
    '⠯': 'and', '⠿': 'for', '⠷': 'the', '⠮': 'this', '⠾': 'with',

    # Space
    '⠀': ' ',
}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def contains_braille(text):
    """Check if text contains Braille patterns (Unicode range 2800-28FF)"""
    for char in text:
        if '\u2800' <= char <= '\u28ff':
            return True
    return False


def extract_text_from_pdf(filepath):
    """Extract text from PDF, handling both text and image-based PDFs"""
    text = ""

    # First try reading as text PDF
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"PDF text extraction error: {e}")

    # If no text found, try OCR (for image-based PDFs)
    if not text.strip():
        try:
            images = convert_pdf_to_images(filepath)
            for img in images:
                text += pytesseract.image_to_string(img, config='--psm 6')
        except Exception as e:
            print(f"OCR error: {e}")

    return text


def convert_pdf_to_images(filepath):
    """Convert PDF pages to images (simplified version)"""
    images = []
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                # This is a simplified approach - in production use pdf2image or similar
                # For demo purposes, we'll just return an empty list
                pass
    except Exception as e:
        print(f"PDF to image conversion error: {e}")
    return images


def translate_braille(text, language='en'):
    """Convert Braille Unicode characters to specified language"""
    translated = []
    i = 0
    while i < len(text):
        char = text[i]

        # Handle numbers (if next char is a number after ⠼)
        if char == '⠼' and i + 1 < len(text):
            next_char = text[i + 1]
            num_key = f'⠼{next_char}'
            if num_key in braille_to_text:
                translated.append(braille_to_text[num_key])
                i += 2
                continue

        # Handle capital letters (if next char is a letter after ⠠)
        elif char == '⠠' and i + 1 < len(text):
            next_char = text[i + 1]
            if next_char in braille_to_text and next_char.isalpha():
                translated.append(braille_to_text[next_char].upper())
                i += 2
                continue

        # Default case
        translated.append(braille_to_text.get(char, char))
        i += 1

    return ''.join(translated)


# ... [Keep all your existing routes: home, logs, login, signups, signup, logout] ...

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        flash('Please log in first', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']
        language = request.form.get('language', 'en')

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Only PDF files are allowed', 'error')
            return redirect(request.url)

        try:
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract text from PDF
            text = extract_text_from_pdf(filepath)

            if not text.strip():
                flash('The PDF appears to be empty or could not be read', 'error')
                return redirect(request.url)

            # Check for Braille characters
            if not contains_braille(text):
                flash('The PDF does not contain Braille characters', 'error')
                return redirect(request.url)

            # Translate Braille to text
            translated = translate_braille(text, language)

            # Store results in session
            session['original_text'] = text[:1000] + "..." if len(text) > 1000 else text
            session['translated_text'] = translated[:1000] + "..." if len(translated) > 1000 else translated
            session['translation_language'] = language

            flash('File processed successfully!', 'success')
            return redirect(url_for('dashboard'))

        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(request.url)

    return render_template('dashboard.html',
                           username=session.get('username'),
                           original_text=session.get('original_text'),
                           translated_text=session.get('translated_text'),
                           language=session.get('translation_language', 'en'))


if __name__ == '__main__':
    app.run(debug=True)