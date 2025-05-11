from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from werkzeug.utils import secure_filename
import PyPDF2
import pytesseract
from docx import Document
from deep_translator import GoogleTranslator


app = Flask(__name__)
app.secret_key = "The APP key"

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


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login',methods=['GET','POST'])
def logs():
    return render_template('login.html')

@app.route('/log', methods=['POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['email']=email
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')

    return render_template('login.html')


@app.route('/signups')
def sign():
    return render_template('signup.html')
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Basic validation
        if len(username) < 3:
            flash('Username must be at least 3 characters', 'error')
            return redirect(url_for('signup'))
        if len(password) < 8:
            flash('Password must be at least 8 characters', 'error')
            return redirect(url_for('signup'))
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ? OR email = ?', (username, email))
        existing_user = c.fetchone()

        if existing_user:
            conn.close()
            flash('Username or email already exists', 'error')
            return redirect(url_for('signup'))

        # Hash password and store user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                  (username, email, hashed_password))
        conn.commit()
        conn.close()

        flash('Account created successfully! Please log in.', 'success')
        return redirect('/login')

    return render_template('signup.html')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'doc', 'docx'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

def convert_pdf_to_images(filepath):
    # In a production environment, you would use pdf2image or similar
    # For this example, we'll return an empty list
    return []
def contains_braille(text):
    for char in text:
        if '\u2800' <= char <= '\u28ff':
            return True
    return False


def extract_text_from_file(filepath, extension):
    text = ""
    try:
        if extension == 'pdf':
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""

            # If no text found, try OCR (for image-based PDFs)
            if not text.strip():
                try:
                    images = convert_pdf_to_images(filepath)
                    for img in images:
                        text += pytesseract.image_to_string(img, config='--psm 6')
                except Exception as e:
                    print(f"OCR error: {e}")

        elif extension == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

        elif extension in ['doc', 'docx']:
            doc = Document(filepath)
            for para in doc.paragraphs:
                text += para.text + '\n'

    except Exception as e:
        print(f"Error extracting text from {extension}: {e}")
    return text


def translate_braille(text, language):
    translated = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == '⠼' and i + 1 < len(text):  # Number prefix
            next_char = text[i + 1]
            num_key = f'⠼{next_char}'
            if num_key in braille_to_text:
                translated.append(braille_to_text[num_key])
                i += 2
                continue
        elif char == '⠠' and i + 1 < len(text):  # Capital letter prefix
            next_char = text[i + 1]
            if next_char in braille_to_text and next_char.isalpha():
                translated.append(braille_to_text[next_char].upper())
                i += 2
                continue
        translated.append(braille_to_text.get(char, char))
        i += 1
    base_translation = ''.join(translated)
    if language.lower() != 'english':
        try:
            return GoogleTranslator(source='auto', target=language.lower()).translate(base_translation)
        except:
            return base_translation
    return base_translation

@app.route('/dashboard', methods=['GET'])
def dashboard():
    if 'user_id' not in session:
        flash('Please log in first', 'error')
        return redirect(url_for('login'))
    return render_template('dashboard.html',
                           username=session.get('username'),
                           original_text=session.get('original_text'),
                           translated_text=session.get('translated_text'),
                           language=session.get('translation_language', 'en'))


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user_id' not in session:
        flash('Please log in first', 'error')
        return redirect(url_for('login'))

    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('dashboard'))

    file = request.files['file']
    language = request.form.get('language', 'en')

    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('dashboard'))

    if not allowed_file(file.filename):
        flash('Allowed file types are PDF, TXT, DOC, and DOCX', 'error')
        return redirect(url_for('dashboard'))

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        extension = filename.rsplit('.', 1)[1].lower()
        text = extract_text_from_file(filepath, extension)

        if not text.strip():
            flash('The file appears to be empty or could not be read', 'error')
            return redirect(url_for('dashboard'))

        if not contains_braille(text):
            flash('The file does not contain Braille characters', 'error')
            return redirect(url_for('dashboard'))

        translated = translate_braille(text, language)

        # Store results in session (limited to 1000 chars to prevent session size issues)
        session['original_text'] = text[:1000] + "..." if len(text) > 1000 else text
        session['translated_text'] = translated[:1000] + "..." if len(translated) > 1000 else translated
        session['translation_language'] = language

        flash('File processed successfully!', 'success')
        return redirect(url_for('dashboard'))

    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)