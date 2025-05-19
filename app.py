import os
from flask import Flask, render_template, request, jsonify
import tempfile
from werkzeug.utils import secure_filename
from modules.pdf_processor import process_pdf
from modules.question_gen import QuestionGenerator
from modules.answer_gen import AnswerGenerator
import pypdf  # Added for PDF processing

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16MB

# Clés API pour les services
GROQ_API_KEY = "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"
HUGGINGFACE_API_KEY = "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"

# Initialisation des générateurs
question_gen = QuestionGenerator(GROQ_API_KEY)

# Créer le dossier de documents s'il n'existe pas
DOCUMENTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# Initialisation du générateur de réponses (avec lazy loading)
answer_generator = None

def get_answer_generator():
    global answer_generator
    if answer_generator is None:
        print("Initialisation du générateur de réponses...")
        answer_generator = AnswerGenerator(documents_folder=DOCUMENTS_FOLDER, api_key=HUGGINGFACE_API_KEY)
    return answer_generator

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Enregistrer aussi dans le dossier documents/ pour la RAG
        document_path = os.path.join(DOCUMENTS_FOLDER, filename)
        file.seek(0)  # Réinitialiser le pointeur de fichier
        try:
            with open(document_path, 'wb') as f:
                f.write(file.read())
            print(f"Document enregistré pour la RAG: {document_path}")
            
            # Update vector store with new document
            generator = get_answer_generator()
            success = generator.update_vectorstore_with_document(document_path)
            if success:
                print(f"Document added to vector store successfully: {filename}")
            else:
                print(f"Failed to add document to vector store: {filename}")
        except Exception as e:
            print(f"Erreur lors de l'enregistrement du document pour la RAG: {e}")
        
        # Traitement du PDF
        try:
            sections = process_pdf(filepath)
            
            # Génération des questions pour chaque section
            result = []
            for title, content in sections:
                if len(content.split()) < 40:  # Ignorer les sections trop courtes
                    continue
                
                questions = question_gen.generate_questions(content, 5)
                categories = question_gen.categorize_questions(questions)
                
                if categories:
                    result.append({
                        'title': title,
                        'categories': categories
                    })
            
            # Nettoyer le fichier temporaire - mais vérifier d'abord s'il existe
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as clean_error:
                print(f"Warning: Could not delete temporary file: {clean_error}")
            
            return jsonify({'sections': result})
        
        except Exception as e:
            # Ne pas supprimer le fichier en cas d'erreur, cela pourrait causer des problèmes
            print(f"Error processing PDF: {e}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File must be a PDF'}), 400

@app.route('/answer', methods=['POST'])
def generate_answer():
    """Endpoint pour générer une réponse à une question"""
    if not request.json or 'question' not in request.json:
        return jsonify({'error': 'La question est manquante'}), 400
    
    question = request.json['question']
    
    try:
        # Lazy-loading du générateur de réponses
        generator = get_answer_generator()
        
        # Générer la réponse
        answer = generator.generate_answer(question)
        
        return jsonify({'answer': answer})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 