"""
Motor RAG para documentos legales - VERSION CORREGIDA
"""

import pickle
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path

# Para PDFs
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("ERROR: PyPDF2 no instalado. Ejecuta: pip install PyPDF2")

# Para embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("ERROR: sentence-transformers no instalado")

# Para búsqueda vectorial
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("ERROR: faiss-cpu no instalado")

import warnings
warnings.filterwarnings('ignore')


class RAGLegalSystem:
    """Sistema RAG completo sin dependencias de langchain"""
    
    def __init__(self, chunk_size: int = 500, top_k: int = 5):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.chunks = []
        self.metadata = []
        self.raw_documents = []
        self.index = None
        self.is_loaded = False
        
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers no instalado")
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu no instalado")
        
        print("Cargando modelo de embeddings...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Modelo cargado")
    
    def load_pdfs(self, pdf_folder: str = "data"):
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 no instalado")
        
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No hay PDFs en '{pdf_folder}'")
        
        print(f"Encontrados {len(pdf_files)} PDFs")
        self.raw_documents = []
        
        for pdf_path in pdf_files:
            try:
                print(f"  Leyendo: {pdf_path.name}")
                reader = PdfReader(str(pdf_path))
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        self.raw_documents.append({
                            "text": text,
                            "source": pdf_path.name,
                            "page": page_num
                        })
            except Exception as e:
                print(f"  Error con {pdf_path.name}: {e}")
        
        print(f"Cargadas {len(self.raw_documents)} paginas")
        return {"total_pages": len(self.raw_documents)}
    
    def process_chunks(self):
        if not self.raw_documents:
            raise ValueError("Primero carga PDFs con load_pdfs()")
        
        print(f"Creando chunks de {self.chunk_size} caracteres...")
        self.chunks = []
        self.metadata = []
        
        for doc in self.raw_documents:
            text = doc["text"]
            step = self.chunk_size - 100
            
            for i in range(0, len(text), step):
                chunk = text[i:i + self.chunk_size]
                if len(chunk) > 50:
                    self.chunks.append(chunk)
                    self.metadata.append({
                        "source": doc["source"],
                        "page": doc["page"]
                    })
        
        print(f"Generados {len(self.chunks)} fragmentos")
        return {"total_chunks": len(self.chunks)}
    
    def build_index(self):
        if not self.chunks:
            raise ValueError("Primero procesa chunks con process_chunks()")
        
        print("Generando embeddings...")
        embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        embeddings = embeddings.astype('float32')
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        self.is_loaded = True
        print(f"Indice creado con {self.index.ntotal} vectores")
        return {"dimension": dimension}
    
    def search(self, query: str, k: int = None):
        if not self.is_loaded:
            raise ValueError("Indice no construido")
        
        k = k or self.top_k
        k = min(k, len(self.chunks))
        
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "score": float(1 / (1 + dist))
                })
        return results
    
    def answer_question(self, query: str):
        results = self.search(query)
        
        if not results:
            return {
                "pregunta": query,
                "respuesta_llm": "No se encontró información relevante.",
                "fragmentos_contexto": [],
                "fuentes": [],
                "num_fragmentos_usados": 0
            }
        
        # Construir respuesta - CORREGIDO: sin f-string mal formada
        respuesta = "**Respuesta basada en " + str(len(results)) + " fragmentos encontrados:**\n\n"
        
        for i, r in enumerate(results[:3], 1):
            texto = r['text'][:300]
            if len(r['text']) > 300:
                texto = texto + "..."
            
            respuesta = respuesta + str(i) + ". " + texto + "\n"
            respuesta = respuesta + "   Fuente: " + r['metadata']['source'] + " (pag. " + str(r['metadata']['page']) + ")\n\n"
        
        fuentes = list(set([r['metadata']['source'] for r in results]))
        
        return {
            "pregunta": query,
            "respuesta_llm": respuesta,
            "fragmentos_contexto": [r['text'] for r in results[:3]],
            "fuentes": fuentes,
            "num_fragmentos_usados": len(results)
        }
    
    def save_index(self, path: str = "rag_index.pkl"):
        """Guarda el índice en disco"""
        data = {
            'chunks': self.chunks,
            'metadata': self.metadata,
            'index': self.index,
            'chunk_size': self.chunk_size,
            'top_k': self.top_k
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Indice guardado en {path}")
    
    def load_index(self, path: str = "rag_index.pkl"):
        """Carga un índice guardado"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.metadata = data['metadata']
        self.index = data['index']
        self.chunk_size = data['chunk_size']
        self.top_k = data['top_k']
        self.is_loaded = True
        
        print(f"Indice cargado: {len(self.chunks)} fragmentos")