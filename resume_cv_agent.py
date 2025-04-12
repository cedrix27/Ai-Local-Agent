import fitz  # PyMuPDF
import gradio as gr
from langchain_community.llms import Ollama 
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# ⚙️ Modèle Llama3 via Ollama
llm = Ollama(model="llama3:8b")

# 📄 Template pour structurer le résumé du CV
template = """
Voici le contenu brut d’un CV :

{cv_texte}

Résume ce CV de manière claire pour un recruteur, en extrayant :
- Nom complet (si présent)
- Poste recherché
- Compétences techniques
- Expériences principales
- Diplômes obtenus
- Langues maîtrisées
- Outils ou technos connus

Formate tout proprement.
"""

prompt = PromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt)

# 🔍 Fonction d’extraction de texte depuis un PDF
def extraire_texte_pdf(file):
    with fitz.open(file.name) as doc:
        texte = ""
        for page in doc:
            texte += page.get_text()
    return texte

# 🔁 Pipeline complet
def analyser_cv(file):
    texte = extraire_texte_pdf(file)
    if not texte.strip():
        return "Aucun texte extrait du CV. Vérifie que le PDF n'est pas scanné comme image."

    try:
        resume = chain.run(cv_texte=texte)
        return resume if resume else "Le modèle n'a pas généré de résumé."
    except Exception as e:
        return f"Erreur lors de l'analyse du CV : {str(e)}"


# 🎨 Interface Gradio
gr.Interface(
    fn=analyser_cv,
    inputs=gr.File(label="Upload ton CV (PDF)", file_types=[".pdf"]),
    outputs=gr.Textbox(label="Résumé du CV"),
    title="Agent AI - Résumeur de CV",
    description="Ce module lit et résume un CV PDF pour un recruteur"
).launch()
