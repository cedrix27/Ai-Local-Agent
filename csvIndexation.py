from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document


# Charger ton dataset
df = pd.read_csv("offres_emploi_maroc.csv")

# Créer une colonne "texte" combinant les champs utiles
df["texte"] = df.apply(lambda row: f"{row['titre']} - {row['description']} à {row['ville']} ({row['domaine']}, {row['contrat']}, {row['niveau_etude']})", axis=1)

# Convertir en documents
from langchain.schema.document import Document
documents = [Document(page_content=txt, metadata={"id": i}) for i, txt in enumerate(df["texte"])]

# Embeddings avec Ollama
embedding = OllamaEmbeddings(model="llama3:8b")

# Vectorstore Chroma
vectordb = Chroma.from_documents(documents, embedding, persist_directory="chroma_db")
vectordb.persist()
print("Indexation terminée et sauvegardée dans 'chroma_db'.")