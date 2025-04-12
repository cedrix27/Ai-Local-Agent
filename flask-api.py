from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama

# LLM
llm = Ollama(model="llama3:8b")

# Flask app
app = Flask(__name__)

# 📋 Offre d’emploi prédéfinie
offre_emploi = """
Offre d'emploi : Développeur Fullstack React.js & Node.js
- Développement frontend avec React.js
- Développement backend avec Node.js (Express)
- Intégration d’API REST
- Travail en équipe avec Git
- Maîtrise de MongoDB et/ou PostgreSQL
- Expérience avec Docker est un plus
- Environnement Agile / Scrum
"""

# Fonction de traitement du CV
def extraire_texte_cv(file):
    reader = PdfReader(file)
    texte = ""
    for page in reader.pages:
        texte += page.extract_text() + "\n"
    return texte

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
 Formate tout proprement.

 et donne un pourcentage approximatif de correspondance avec l'offre
 - Pourcentage de correspondance : ...%
 En francais
    """
    reponse = llm.invoke(prompt)
    return reponse

@app.route('/resumer-cv', methods=['POST'])
def resumer_cv():
    if 'cv' not in request.files or 'offre' not in request.form:
        return jsonify({'error': 'CV et offre requis'}), 400

    cv_file = request.files['cv']
    offre = request.form['offre']
    resultat = analyser_cv(cv_file, offre)
    return jsonify({'analyse': resultat})