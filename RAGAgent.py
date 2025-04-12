from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import gradio as gr

# ✅ Embedding correct
embedding = OllamaEmbeddings(model="llama3:8b")

# ✅ Rechargement du vectordb
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vectordb.as_retriever()

# ✅ LLM
llm = Ollama(model="llama3:8b")

# ✅ RAG Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ✅ Fonction de recherche
def rechercher_offres(ville, domaine, contrat, niveau):
    requete = (
        f"Offre d'emploi en {domaine} pour un contrat {contrat}, niveau {niveau}, à {ville}. "
        "Même approximatif(ville differente, pas exactement le meme niveau), proposer les 10 offres les plus proches en priorité. Tu dois répondre en français et sans explication, juste les offres, mais aussi tu dois obligatoirement proproser 10 offres dans le domaines demandé, même si tu n'as pas trouvé d'offres dans la ville demandée. "
    )
    resultat = qa_chain.run(requete)
    return resultat

# ✅ Interface Gradio
iface = gr.Interface(
    fn=rechercher_offres,
    inputs=[
        gr.Textbox(label="Ville"),
        gr.Textbox(label="Domaine d'étude"),
        gr.Textbox(label="Type de contrat (ex: stage, cdi)"),
        gr.Textbox(label="Niveau d'études (ex: Bac+3)")
    ],
    outputs=gr.Textbox(label="Offres recommandées")
)

iface.launch(share=False)
