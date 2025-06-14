from flask import Flask, request, render_template
from PyPDF2 import PdfReader
from langchain_ollama import OllamaLLM

# Initialisation Flask
app = Flask(__name__, static_folder="static")

# LLM via Langchain + Ollama
llm = OllamaLLM(model="llama3:8b")

# Offre d’emploi prédéfinie
offre_emploi_defaut = """
Offre d'emploi : Développeur Fullstack React.js & Node.js
- Développement frontend avec React.js
- Développement backend avec Node.js (Express)
- Intégration d’API REST
- Travail en équipe avec Git
- Maîtrise de MongoDB et/ou PostgreSQL
- Expérience avec Docker est un plus
- Environnement Agile / Scrum
"""

# Extraction de texte depuis le PDF
def extraire_texte_cv(file):
    reader = PdfReader(file)
    texte = ""
    for page in reader.pages:
        texte += page.extract_text() + "\n"
    return texte

# Analyse via LLM
def analyser_cv(cv_file, offre):
    texte_cv = extraire_texte_cv(cv_file)
    prompt = f"""
Voici le descriptif d'une offre d'emploi :
{offre}

Et voici un CV brut :
{texte_cv}

Résume ce CV de manière claire pour un recruteur, en extrayant :
- Nom complet (si présent)
- Poste recherché
- Compétences techniques
- Expériences principales
- Diplômes obtenus
- Langues maîtrisées
- Outils ou technos connus

Formate tout proprement en français.
Donne un pourcentage approximatif de correspondance avec l'offre :
- Pourcentage de correspondance : ...%

    """
    reponse = llm.invoke(prompt)
    return reponse

# Interface HTML principale
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Traitement formulaire
@app.route('/resumer-cv', methods=['POST'])
def resumer_cv():
    if 'cv' not in request.files:
        return "Aucun fichier CV envoyé", 400

    cv_file = request.files['cv']
    offre = request.form.get('offre') or offre_emploi_defaut
    resultat = analyser_cv(cv_file, offre)
    return render_template('resultat.html', analyse=resultat)

# Lancement serveur
if __name__ == '__main__':
    app.run(debug=True)
