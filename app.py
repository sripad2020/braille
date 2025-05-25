from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from werkzeug.utils import secure_filename
import PyPDF2
import pytesseract
from docx import Document
from deep_translator import GoogleTranslator
import google.generativeai as genai
from PIL import Image
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re, cv2
import logging
import numpy as np
import base64

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure Gemini API
genai.configure(
    api_key=os.getenv('GEMINI_API_KEY', 'AIzaSyBvph-JgoPgpF51Fb-0Q-9ikeVwaaCTE2A'))  # Use env variable in production

app = Flask(__name__)
app.secret_key = "The APP key"

# Configure upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'doc', 'docx', 'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Braille translation mappings
braille_to_text = {
    '⠁': 'a', '⠃': 'b', '⠉': 'c', '⠙': 'd', '⠑': 'e',
    '⠋': 'f', '⠛': 'g', '⠓': 'h', '⠊': 'i', '⠚': 'j',
    '⠅': 'k', '⠇': 'l', '⠍': 'm', '⠝': 'n', '⠕': 'o',
    '⠏': 'p', '⠟': 'q', '⠗': 'r', '⠎': 's', '⠞': 't',
    '⠥': 'u', '⠧': 'v', '⠺': 'w', '⠭': 'x', '⠽': 'y', '⠵': 'z',
    '⠼⠁': '1', '⠼⠃': '2', '⠼⠉': '3', '⠼⠙': '4', '⠼⠑': '5',
    '⠼⠋': '6', '⠼⠛': '7', '⠼⠓': '8', '⠼⠊': '9', '⠼⠚': '0',
    '⠂': ',', '⠲': '.', '⠆': ';', '⠤': '-',
    '⠦': '?', '⠖': '!', '⠶': '"', '⠄': "'",
    '⠯': 'and', '⠿': 'for', '⠷': 'the', '⠮': 'this', '⠾': 'with',
    '⠀': ' ',
}


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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def convert_pdf_to_images(filepath):
    # Placeholder: Use pdf2image in production
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
            if not text.strip():
                try:
                    images = convert_pdf_to_images(filepath)
                    for img in images:
                        text += pytesseract.image_to_string(img, config='--psm 6')
                except Exception as e:
                    logging.error(f"OCR error: {e}")
        elif extension == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        elif extension in ['doc', 'docx']:
            doc = Document(filepath)
            for para in doc.paragraphs:
                text += para.text + '\n'
    except Exception as e:
        logging.error(f"Error extracting text from {extension}: {e}")
    return text


def analyze_image_with_gemini(filepath):
    try:
        img = Image.open(filepath)

        # Convert image to base64 for display
        with open(filepath, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = [
            img,
                "Detect the image as braille image or not , if Yes analyse the image and indetify characters based on your analysis and explain about the image,"
                "Analyse the dots and give me characters information in English"
        ]
        response = model.generate_content(prompt)

        return {
            'image_base64': encoded_image,
            'analysis': clean_markdown(response.text)
        }
    except Exception as e:
        logging.error(f"Gemini image analysis error: {e}")
        return {
            'image_base64': None,
            'analysis': f"Error analyzing image: {str(e)}"
        }


def convert_paragraph_to_points(paragraph, num_points=5):
    sentences = sent_tokenize(paragraph)
    words = word_tokenize(paragraph.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    freq_dist = FreqDist(filtered_words)
    sentence_scores = {}
    for sentence in sentences:
        sentence_word_tokens = word_tokenize(sentence.lower())
        sentence_word_tokens = [word for word in sentence_word_tokens if word.isalnum()]
        score = sum(freq_dist.get(word, 0) for word in sentence_word_tokens)
        sentence_scores[sentence] = score
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    return sorted_sentences[:min(num_points, len(sorted_sentences))]


def clean_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'`{3}.*?`{3}', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'^\s*>+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\*\-+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def translate_braille(text, language):
    translated = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == '⠼' and i + 1 < len(text):
            next_char = text[i + 1]
            num_key = f'⠼{next_char}'
            if num_key in braille_to_text:
                translated.append(braille_to_text[num_key])
                i += 2
                continue
        elif char == '⠠' and i + 1 < len(text):
            next_char = text[i + 1]
            if next_char in braille_to_text and braille_to_text[next_char].isalpha():
                translated.append(braille_to_text[next_char].upper())
                i += 2
                continue
        translated.append(braille_to_text.get(char, char))
        i += 1
    base_translation = ''.join(translated)
    if language.lower() != 'english':
        try:
            return GoogleTranslator(source='auto', target=language.lower()).translate(base_translation)
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return base_translation
    return base_translation


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
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
            session['email'] = email
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
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                  (username, email, hashed_password))
        conn.commit()
        conn.close()
        flash('Account created successfully! Please log in.', 'success')
        return redirect('/login')
    return render_template('signup.html')


@app.route('/dashboard', methods=['GET'])
def dashboard():
    if 'user_id' not in session:
        flash('Please log in first', 'error')
        return redirect(url_for('logs'))
    return render_template('dashboard.html',
                           username=session.get('username'),
                           original_text=session.get('original_text'),
                           translated_text=session.get('translated_text'),
                           language=session.get('translation_language', 'en'),
                           tone_points=session.get('tone_points'),
                           uploaded_image=session.get('uploaded_image'),
                           is_image=session.get('is_image', False))


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
        flash('Allowed file types are PDF, TXT, DOC, DOCX, PNG, JPG, JPEG', 'error')
        return redirect(url_for('dashboard'))
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        extension = filename.rsplit('.', 1)[1].lower()

        # Clear previous session data
        session.pop('original_text', None)
        session.pop('translated_text', None)
        session.pop('tone_points', None)
        session.pop('uploaded_image', None)
        session.pop('is_image', None)

        if extension in ['png', 'jpg', 'jpeg']:
            # Handle image upload
            analysis_result = analyze_image_with_gemini(filepath)
            session['original_text'] = None
            session['translated_text'] = analysis_result['analysis']
            session['uploaded_image'] = analysis_result['image_base64']
            session['is_image'] = True
            flash('Image analyzed successfully!', 'success')
        else:
            # Handle document upload
            text = extract_text_from_file(filepath, extension)
            if not text.strip():
                flash('The file appears to be empty or could not be read', 'error')
                return redirect(url_for('dashboard'))
            if not contains_braille(text):
                flash('The file does not contain Braille characters', 'error')
                return redirect(url_for('dashboard'))
            translated = translate_braille(text, language)
            session['original_text'] = text[:1000] + "..." if len(text) > 1000 else text
            session['translated_text'] = translated[:1000] + "..." if len(translated) > 1000 else translated
            session['translation_language'] = language
            session['is_image'] = False
            flash('File processed successfully!', 'success')

        return redirect(url_for('dashboard'))
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('dashboard'))


@app.route('/tone_analysis', methods=['GET'])
def tone_analysis():
    if 'user_id' not in session:
        flash('Please log in first', 'error')
        return redirect(url_for('login'))
    if not session.get('translated_text'):
        flash('No translated text available. Please upload a file first.', 'error')
        return redirect(url_for('dashboard'))
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Analyze the tone of the following text and describe its emotional sentiment (e.g., positive, negative, neutral, etc.):
        "{session['translated_text'][:1000]}"

        Provide:
        1. The overall tone of the text (e.g., positive, negative, neutral).
        2. Key indicators in the text that suggest this tone.
        3. Potential contexts where this tone might be used.
        4. Any nuances or mixed tones present.

        Format the response in clear bullet points.
        """
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 500})
        tone_analysis = clean_markdown(response.text)
        tone_points = convert_paragraph_to_points(tone_analysis, num_points=5)
        session['tone_points'] = tone_points
        flash('Tone analysis completed successfully!', 'success')
    except Exception as e:
        logging.error(f"Gemini tone analysis error: {e}")
        session['tone_points'] = [f"Could not perform tone analysis: {str(e)}"]
        flash('Error performing tone analysis.', 'error')
    return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('logs'))


if __name__ == '__main__':
    app.run()