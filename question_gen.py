import os
import regex as re2
from openai import OpenAI
from huggingface_hub import login
from getpass import getpass

LABELS = ["AML / KYC", "Fund Regulation", "Market Manipulation",
          "Stablecoins", "Custody & Wallet Security",
          "Liquidity", "Tokenomics", "Tax & Reporting"]

def authenticate_huggingface():
    token = getpass("🔐 Enter your Hugging Face API token: ")
    login(token=token)
    print("✅ Logged into Hugging Face")

class QuestionGenerator:
    def __init__(self, api_key=None):
        # Si une clé API est fournie, utiliser Groq
        if api_key:
            # Configuration de l'API Groq
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
            
            # Initialisation du client sans arguments problématiques
            try:
                self.client = OpenAI()
            except TypeError:
                # Solution alternative si la première méthode échoue
                import httpx
                # Créer un client httpx personnalisé sans proxies
                http_client = httpx.Client(
                    base_url="https://api.groq.com/openai/v1",
                    timeout=60.0,
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                self.client = OpenAI(api_key=api_key, http_client=http_client)
            self.use_huggingface = False
        else:
            # Utiliser Hugging Face
            authenticate_huggingface()
            self.use_huggingface = True
            
            # Import des bibliothèques nécessaires pour Hugging Face
            from transformers import pipeline
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Chargement des modèles
            self.generator = pipeline(
                "text2text-generation", 
                model="google/flan-t5-small",
                device=self.device
            )
            
            self.classifier = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli",
                device=self.device
            )
            
            print(f"Modèles Hugging Face chargés avec succès sur {self.device}")
    
    def generate_questions(self, passage, k=1):
        """Génère une seule question à partir d'un passage de texte"""
        if not self.use_huggingface:
            # Utilisation de l'API Groq/OpenAI
            prompt = ("Act like a senior compliance officer, understand deeply this document "
                     "and generate the most critical question that I need to ask to generate a due diligence report:\n\n"
                     f"{passage}\n\nPlease provide 1 critical question.")

            try:
                rsp = self.client.chat.completions.create(
                    model="gemma2-9b-it",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=100, temperature=0.7
                )
                raw = rsp.choices[0].message.content
                question = re2.sub(r"^[\-\d\.\)\s]*","", raw.strip())
                question = question.rstrip(".")+"?" if not question.endswith("?") else question
                return [question] if question else []
            except Exception as e:
                print(f"Error generating question: {e}")
                return []
        else:
            # Utilisation des modèles Hugging Face
            prompt = f"Generate a critical compliance question for due diligence report based on this document: {passage}"
            
            try:
                result = self.generator(
                    prompt,
                    max_length=100,
                    temperature=0.7,
                    num_return_sequences=1
                )
                
                question = result[0]['generated_text'].strip()
                # S'assurer que c'est une question
                question = re2.sub(r"^[\-\d\.\)\s]*", "", question)
                question = question.rstrip(".")+"?" if not question.endswith("?") else question
                
                return [question] if question else []
            except Exception as e:
                print(f"Error generating question with Hugging Face: {e}")
                return []
    
    def categorize_questions(self, questions):
        """Catégorise la question générée"""
        if not questions: return {}
        results = {l:[] for l in LABELS}

        # On ne travaille qu'avec la première question
        q = questions[0]
        
        if not self.use_huggingface:
            # Utilisation de l'API Groq/OpenAI
            prompt = (f"Act like a senior compliance officer. Choose ONLY ONE of these categories for the following question: "
                     f"{', '.join(LABELS)}.\n\nQuestion: {q}\n\n"
                     f"Reply with just the category name, nothing else.")

            try:
                rsp = self.client.chat.completions.create(
                    model="gemma2-9b-it",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0,
                    max_tokens=50
                )

                category = rsp.choices[0].message.content.strip()

                # Find the best matching category
                matched_category = None
                for label in LABELS:
                    if label.lower() in category.lower():
                        matched_category = label
                        break

                # If no match found, use the closest match
                if not matched_category:
                    for label in LABELS:
                        if any(word.lower() in category.lower() for word in label.split()):
                            matched_category = label
                            break

                # If still no match, assign to first category
                if not matched_category and LABELS:
                    matched_category = LABELS[0]

                # Add question to the matched category
                if matched_category:
                    results[matched_category].append(q)

            except Exception as e:
                print(f"Error processing question: {q}")
                print(f"Error: {e}")
                # Pour le prototype, en cas d'erreur, assignation au premier label
                if LABELS:
                    results[LABELS[0]].append(q)
        else:
            # Utilisation des modèles Hugging Face
            try:
                # Utilisation de zero-shot classification pour catégoriser la question
                classification = self.classifier(
                    q, 
                    candidate_labels=LABELS,
                    hypothesis_template="This question is about {}."
                )
                
                # Récupérer la catégorie avec le score le plus élevé
                best_category = classification['labels'][0]
                results[best_category].append(q)
                
            except Exception as e:
                print(f"Error processing question with Hugging Face: {q}")
                print(f"Error: {e}")
                # En cas d'erreur, assignation au premier label
                if LABELS:
                    results[LABELS[0]].append(q)
        
        return {k:v for k,v in results.items() if v} 