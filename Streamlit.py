import streamlit as st
from PyPDF2 import PdfReader
from langchain_ollama import OllamaLLM

# LLM via Langchain + Ollama
llm = OllamaLLM(model="llama3:8b")

# Offre d'emploi par défaut
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

# Extraction du texte du CV
def extraire_texte_cv(file):
    reader = PdfReader(file)
    texte = ""
    for page in reader.pages:
        texte += page.extract_text() + "\n"
    return texte

# Analyse du CV
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

# -----------------------------
# Application Streamlit
# -----------------------------

st.set_page_config(page_title="Résumé de CV & Matching", layout="centered")
st.title("📄 Résumé de CV & Matching avec Offre d'Emploi")

uploaded_file = st.file_uploader("Upload ton CV en PDF", type=["pdf"])
offre = st.text_area("Colle ici la description de l'offre d'emploi (ou laisse vide pour utiliser l'offre par défaut)", height=200)

if st.button("Analyser le CV"):
    if uploaded_file is not None:
        offre_utilisee = offre if offre.strip() != "" else offre_emploi_defaut
        with st.spinner('Analyse en cours...'):
            resultat = analyser_cv(uploaded_file, offre_utilisee)
        
        st.subheader("👍 Résultat de l'analyse :")
        st.write(resultat)
    else:
        st.warning("Merci d'uploader un fichier CV en PDF.")
