"""
Aplicación RAG Legal con Streamlit
"""

import streamlit as st
import time
from pathlib import Path

# Importar nuestro RAG
from rag_engine import RAGLegalSystem

# Configuración de la página
st.set_page_config(
    page_title="Sistema RAG Legal",
    page_icon="⚖️",
    layout="wide"
)

# Estilos CSS
st.markdown("""
<style>
    .response-box {
        background-color: #f0fdf4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #22c55e;
        margin: 1rem 0;
    }
    .main-header {
        font-size: 2rem;
        color: #1f3b4c;
        text-align: center;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">⚖️ Asistente Legal RAG</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuración")
        
        chunk_size = st.select_slider(
            "Tamaño de fragmento",
            options=[300, 500, 800],
            value=500
        )
        
        top_k = st.slider("Fragmentos a recuperar", 1, 10, 5)
        
        st.divider()
        st.header("📁 Documentos")
        
        # Crear carpeta data
        Path("data").mkdir(exist_ok=True)
        
        # Subir PDFs
        uploaded_files = st.file_uploader(
            "Subir PDFs",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                with open(Path("data") / file.name, "wb") as f:
                    f.write(file.getbuffer())
            st.success(f"✅ {len(uploaded_files)} PDFs guardados")
        
        # Mostrar PDFs existentes
        pdfs = list(Path("data").glob("*.pdf"))
        if pdfs:
            st.info(f"📄 {len(pdfs)} PDFs disponibles")
        
        st.divider()
        
        # Botón de inicialización
        if st.button("🚀 Inicializar Sistema", type="primary", use_container_width=True):
            with st.spinner("Inicializando..."):
                try:
                    if not pdfs:
                        st.error("No hay PDFs. Sube al menos uno.")
                        return
                    
                    rag = RAGLegalSystem(chunk_size=chunk_size, top_k=top_k)
                    
                    status = st.empty()
                    status.info("Cargando PDFs...")
                    rag.load_pdfs("data")
                    
                    status.info("Creando fragmentos...")
                    rag.process_chunks()
                    
                    status.info("Construyendo índice...")
                    rag.build_index()
                    
                    status.empty()
                    
                    st.session_state.rag = rag
                    st.session_state.initialized = True
                    
                    st.success(f"✅ Sistema listo! {len(rag.chunks)} fragmentos")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Área de chat
    if st.session_state.get('initialized', False):
        rag = st.session_state.rag
        
        # Estadísticas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fragmentos", len(rag.chunks))
        with col2:
            st.metric("Top-k", rag.top_k)
        with col3:
            st.metric("Chunk size", f"{rag.chunk_size}")
        
        st.divider()
        
        # Chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Input
        if prompt := st.chat_input("Haz una pregunta..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Buscando..."):
                    start = time.time()
                    response = rag.answer_question(prompt)
                    elapsed = time.time() - start
                    
                    st.markdown(f'<div class="response-box">{response["respuesta_llm"]}</div>', 
                               unsafe_allow_html=True)
                    st.caption(f"⏱️ {elapsed:.2f}s | 📄 {response['num_fragmentos_usados']} fragmentos")
                    
                    if response["fuentes"]:
                        with st.expander("📚 Fuentes"):
                            for f in response["fuentes"]:
                                st.write(f"• {f}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["respuesta_llm"]
                })
    else:
        st.info("👈 Configura e inicializa el sistema en la barra lateral")


if __name__ == "__main__":
    main()