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
def authenticate_huggingface():
    login(token=HUGGINGFACE_API_KEY)
    print("✅ Logged into Hugging Face")


def load_model_and_tokenizer(model_name):
    print(f"🔄 Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    print("✅ Model and tokenizer loaded successfully")
    print_gpu_memory()

class BGEEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0]

'''def get_answer_generator():
    global answer_generator
    if answer_generator is None:
        print("Initialisation du générateur de réponses...")
        answer_generator = AnswerGenerator(documents_folder=DOCUMENTS_FOLDER, api_key=HUGGINGFACE_API_KEY)
    return answer_generator*/'''

# ------------------------------
# Step 5: 
# ------------------------------

def build_prompt(user_question, context):
    return f"""
<|system|>
You are a senior compliance officer, highly experienced in the Due Diligence process for crypto asset funds.
Use ONLY the provided documentation context below to answer the user's question.
Keep the answer short, structured, professional, and comprehensive.
If the question is unclear or irrelevant, politely ask for clarification.

Context:
{context}
</|system|>

<|user|>
{user_question}
</|user|>

<|assistant|>
"""

# ------------------------------
# Step 6: Similarity Metrics
# ------------------------------

def compute_similarity(query, docs):
    # ⚠️ Placeholder function
    query_embedding = embedding.embed_query(query)  # Utilisation de embed_query
    doc_embeddings = embedding.embed_documents([doc.page_content for doc in docs])  # Utilisation de embed_documents
    similarities = np.dot(doc_embeddings, query_embedding)
    return similarities

def recall_at_k(similarities, k):
    return sum(similarities[:k]) / sum(similarities) if len(similarities) > 0 else 0

def precision_at_k(similarities, k):
    return sum(1 for s in similarities[:k] if s > 0.5) / k if len(similarities) > 0 else 0

def hit_at_k(similarities, k):
    return int(any(s > 0.7 for s in similarities[:k]))

def mean_reciprocal_rank(similarities):
    for i, s in enumerate(similarities):
        if s > 0.7:
            return 1.0 / (i + 1)
    return 0.0
# ------------------------------
# Step 6: Similarity Metrics
# ------------------------------


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
