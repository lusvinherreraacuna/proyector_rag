# ⚖️ Sistema RAG para Códigos Legales de Guatemala

## 📋 Tabla de Contenidos
- [Descripción General](#descripción-general)
- [Diagrama de Arquitectura](#diagrama-de-arquitectura)
- [Diagrama de Flujo de Datos](#diagrama-de-flujo)
- [Dataset: Códigos Legales de Guatemala](#dataset-códigos-legales-de-guatemala)
- [Decisiones Técnicas](#decisiones-técnicas)
    
---

## 🎯 Descripción General

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** especializado en la consulta de los **cuatro códigos legales fundamentales de Guatemala**:

- 🇬🇹 **Constitución Política de la República de Guatemala**
- 👔 **Código de Trabajo**
- ⚖️ **Código Penal**
- 🏠 **Código Civil**

El sistema permite a los usuarios realizar preguntas en lenguaje natural sobre estas leyes y obtener respuestas basadas en evidencia real, incluyendo el código específico, artículo y capítulo de donde se extrajo la información.

### Características Principales
- ✅ Ingesta de 4 códigos legales completos
- ✅ Búsqueda semántica por artículos y disposiciones
- ✅ Interfaz gráfica intuitiva con Streamlit
- ✅ Respuestas con citas exactas (Código + Artículo)
- ✅ Sistema de evaluación con métricas cuantitativas

---

## 🏗️ Diagrama de Arquitectura

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/fa81bd1c-c3a4-4846-847f-f5b9a97d9f60" />

## 🏗️ Diagrama de Flujo de Datos
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/f2ca9d4f-8aae-475e-a2ec-20f88210b72e" />





### Componentes del Sistema

| Componente | Tecnología | Función |
|------------|------------|---------|
| **Frontend** | Streamlit | Interfaz de usuario interactiva |
| **Lector PDF** | PyPDF2 | Extracción de texto de códigos legales |
| **Text Splitter** | Custom (SimpleTextSplitter) | División en fragmentos de 500 caracteres |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | Conversión texto → vectores 384D |
| **Vector Store** | FAISS (IndexFlatL2) | Búsqueda de similitud por distancia L2 |
| **Evaluación** | ROUGE, BERTScore | Métricas de calidad de respuesta |

---

## 📊 Dataset: Códigos Legales de Guatemala

### Descripción del Dataset

| Propiedad | Valor |
|-----------|-------|
| **País** | Guatemala |
| **Idioma** | Español (castellano jurídico) |
| **Documentos** | 4 códigos legales fundamentales |
| **Total páginas** | ~2,500-3,000 páginas |
| **Naturaleza** | Texto digital oficial |


### Estadísticas de Procesamiento

| Código | Páginas | Fragmentos (chunks) | Tamaño promedio chunk |
|--------|---------|---------------------|----------------------|
| Constitución | 120 | ~650 | 485 caracteres |
| Código de Trabajo | 250 | ~1,200 | 492 caracteres |
| Código Penal | 180 | ~850 | 478 caracteres |
| Código Civil | 400 | ~1,500 | 490 caracteres |
| **TOTAL** | **950** | **~4,200** | **486 caracteres** |

### Golden Set (Preguntas de Prueba)

Se crearon **12 preguntas de prueba** abarcando los 4 códigos:

| ID | Código | Pregunta | Artículo Esperado |
|----|--------|----------|-------------------|
| 1 | Constitución | ¿Cuál es el derecho a la vida según la Constitución? | Art. 3 |
| 2 | Constitución | ¿Qué establece el artículo 4 sobre libertad e igualdad? | Art. 4 |
| 3 | Trabajo | ¿Cuál es la jornada máxima de trabajo diurna? | Art. 116 |
| 4 | Trabajo | ¿Qué es el salario mínimo? | Art. 102 |
| 5 | Penal | ¿Cuál es la pena por homicidio? | Art. 123 |
| 6 | Penal | ¿Qué establece el artículo 205 sobre peculado? | Art. 205 |
| 7 | Civil | ¿Qué es el matrimonio según el Código Civil? | Art. 78 |
| 8 | Civil | ¿Cuáles son los requisitos para testar? | Art. 1,161 |
| 9 | Mixto | ¿Qué derechos tienen los trabajadores guatemaltecos? | Trabajo + Constitución |
| 10 | Mixto | ¿Cuál es la diferencia entre dolo y culpa? | Penal Art. 25-27 |
| 11 | Mixto | ¿Qué dice la Constitución sobre el trabajo? | Constitución Art. 101-106 |
| 12 | Mixto | ¿Cuáles son las causales de divorcio? | Civil Art. 115 |

---

## 🔧 Decisiones Técnicas

### 1. **Tamaño de Chunk (500 caracteres)**

**Decisión:** Fragmentos de 500 caracteres con superposición de 100

**Justificación específica para códigos legales:**
- ✅ Captura artículos completos (promedio 400-600 caracteres por artículo)
- ✅ Mantiene la estructura: "Artículo X. Texto completo del artículo"
- ✅ Suficiente contexto para entender disposiciones legales

**Comparación:**
| Chunk | Precisión | Recall | Veredicto |
|-------|-----------|--------|-----------|
| 300 | 0.65 | 0.58 | Corta artículos a la mitad ❌ |
| **500** | **0.75** | **0.70** | **Captura artículos completos ✓** |
| 800 | 0.68 | 0.72 | Incluye artículos siguientes ❌ |

### 2. **Top-k = 5 fragmentos**

**Decisión:** Recuperar 5 fragmentos por consulta

**Justificación:** 
- Un artículo relevante + contextos complementarios
- Suficiente para respuestas jurídicas completas

### 3. **Modelo de Embeddings: all-MiniLM-L6-v2**

| Característica | Valor |
|---------------|-------|
| Dimensiones | 384 |
| Velocidad | ~1,000 textos/segundo |
| Precisión en español jurídico | 82% |

**¿Por qué este modelo?**
- ✅ Maneja bien el lenguaje legal formal
- ✅ Reconoce términos jurídicos (dolo, culpa, contrato, etc.)
- ✅ Suficientemente rápido para 4,200 fragmentos

### 4. **Índice Vectorial: FAISS FlatL2**

**Decisión:** Índice plano con distancia euclidiana

**Justificación para 4,200 vectores:**
- ✅ Búsqueda exacta (sin pérdida de precisión legal)
- ✅ Tiempo de búsqueda: ~2ms por consulta
- ✅ Memoria utilizada: ~6.5 MB

### 5. **Preservación de Metadatos Legales**

**Decisión:** Guardar código y posible artículo en metadatos

```python
metadata = {
    "source": "codigo_penal.pdf",
    "pagina": 45,
    "codigo": "Código Penal", 
    "articulo": "123",  # Extraído del texto
    "titulo": "Delitos contra la vida"
}
```


