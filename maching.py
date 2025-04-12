import PyPDF2
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import gradio as gr

# 🔍 Fonction pour lire un fichier PDF
def lire_pdf(path):
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        texte = ""
        for page in reader.pages:
            texte += page.extract_text()
    return texte

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

# 🧠 LLM via Ollama
llm = Ollama(model="llama3:8b")

# 🧠 Prompt d’analyse
prompt_template = PromptTemplate.from_template("""
Tu es un assistant RH intelligent. Résume ce CV, extrait les points clés du profil, compétences techniques, technologies utilisées, et expériences importantes. Ensuite, compare-le avec l'offre suivante, puis donne un pourcentage de correspondance (approximation) entre le profil et l’offre d’emploi. Explique les points forts et les points faibles du candidat. En francais

CV :
{cv}

Offre d’emploi :
{offre}

Réponds de manière structurée.
""")

# 📌 Fonction principale
def analyser_cv(cv_path):
    cv_text = lire_pdf(cv_path)
    prompt = prompt_template.format(cv=cv_text, offre=offre_emploi)
    reponse = llm.invoke(prompt)
    return reponse

# 🧪 Interface Gradio
gr.Interface(
    fn=analyser_cv,
    inputs=gr.File(label="Uploader votre CV (PDF)", file_types=[".pdf"]),
    outputs=gr.Textbox(label="Analyse du profil"),
    title="🔍 Analyse CV vs Offre Développeur Fullstack",
    description="Comparez votre CV à une offre d'emploi Fullstack React/Node.js et obtenez un résumé + correspondance."
).launch()
