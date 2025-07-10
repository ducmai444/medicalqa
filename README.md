# Medical Question Answering System

A comprehensive medical question-answering system that leverages the Unified Medical Language System (UMLS) for accurate medical knowledge processing and retrieval.

## 🏥 Overview

This project implements an advanced medical QA system that combines multiple AI/ML techniques to provide accurate answers to medical questions. The system integrates:

- **UMLS API Integration** - Access to comprehensive medical terminology and relationships
- **Cross-Encoder Models** - Advanced relevance scoring for medical concepts
- **First-Order Logic Reasoning** - Intelligent inference of medical relationships
- **Named Entity Recognition** - Medical entity extraction and classification
- **Knowledge Graph Ranking** - Sophisticated ranking algorithms for medical information

## 🚀 Features

### Core Components

- **UMLS API Client** (`umls.py`) - Interface with NIH's UMLS terminology services
- **Cross-Encoder Scoring** (`cross_encoder.py`) - Medical concept relevance scoring using MedCPT
- **FOL Reasoner** (`fol.py`) - First-order logic inference for medical relationships
- **NER System** (`ner.py`) - Medical named entity recognition and classification
- **Knowledge Ranking** (`ranking.py`) - Advanced ranking algorithms for medical information
- **Translation Support** (`translation.py`) - Multi-language medical text processing

### Key Capabilities

- 🔍 **Medical Concept Search** - Find relevant medical terms and concepts
- 🧠 **Intelligent Reasoning** - Infer new medical relationships using logical rules
- 📊 **Relevance Scoring** - Advanced scoring of medical concept relevance
- 🌍 **Multi-language Support** - Process medical text in multiple languages
- 🏷️ **Entity Recognition** - Extract and classify medical entities
- 📈 **Knowledge Ranking** - Rank medical information by relevance and accuracy

## 📋 Requirements

### Python Dependencies

```bash
pip install torch
pip install transformers
pip install sentence-transformers
pip install requests
pip install langdetect
pip install numpy
pip install scikit-learn
pip install networkx
pip install fuzzywuzzy
pip install tqdm
pip install peft
```

### External Services

- **UMLS API Key** - Required for accessing NIH's UMLS terminology services
- **GPU Support** - Recommended for optimal performance with cross-encoder models

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd medicalqa
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up UMLS API key**
   ```python
   # Get your API key from: https://uts.nlm.nih.gov/uts/
   UMLS_API_KEY = "your-api-key-here"
   ```

## 📖 Usage

### Basic Usage

```python
from umls import UMLS_API
from cross_encoder import UMLS_CrossEncoder
from fol import FOLReasoner

# Initialize components
umls_api = UMLS_API("your-api-key")
cross_encoder = UMLS_CrossEncoder()
fol_reasoner = FOLReasoner()

# Search for medical concepts
concepts = umls_api.search_cui("diabetes mellitus")

# Get relationships
relations = umls_api.get_relations("C0011849")  # Diabetes CUI

# Score relevance
scores = cross_encoder.score("diabetes symptoms", ["hyperglycemia", "polyuria"])

# Apply logical reasoning
inferred_relations = fol_reasoner.apply_rules_to_kg(relations)
```

### Advanced Usage

```python
# Named Entity Recognition
from ner import MedicalNER
ner = MedicalNER()
entities = ner.extract_entities("Patient has diabetes and hypertension")

# Knowledge Graph Ranking
from ranking import MedicalRanker
ranker = MedicalRanker()
ranked_results = ranker.rank_concepts("heart disease", candidate_concepts)

# Translation Support
from translation import MedicalTranslator
translator = MedicalTranslator()
translated_text = translator.translate("diabetes", target_lang="es")
```

## 🏗️ Architecture

### Component Overview

```
MedicalQA System
├── UMLS Integration Layer
│   ├── Concept Search
│   ├── Relationship Extraction
│   └── Definition Retrieval
├── AI/ML Processing Layer
│   ├── Cross-Encoder Scoring
│   ├── Named Entity Recognition
│   └── Knowledge Graph Ranking
├── Reasoning Layer
│   ├── First-Order Logic Rules
│   ├── Relationship Inference
│   └── Medical Knowledge Expansion
└── Translation Layer
    ├── Multi-language Support
    └── Medical Terminology Translation
```

### Data Flow

1. **Input Processing** - Medical questions are processed and entities extracted
2. **UMLS Querying** - Relevant medical concepts are retrieved from UMLS
3. **Relevance Scoring** - Cross-encoder models score concept relevance
4. **Logical Reasoning** - FOL rules infer additional relationships
5. **Ranking & Selection** - Advanced algorithms rank and select best answers
6. **Output Generation** - Structured medical answers are generated

## 🔧 Configuration

### UMLS API Configuration

```python
# Configure UMLS API
umls_config = {
    "apikey": "your-api-key",
    "version": "current",  # or specific version
    "language": "ENG"      # or other language codes
}
```

### Model Configuration

```python
# Cross-Encoder Configuration
cross_encoder_config = {
    "model_name": "ncbi/MedCPT-Cross-Encoder",
    "max_length": 512,
    "batch_size": 16,
    "device": "cuda"  # or "cpu"
}
```

## 📊 Performance

### Model Performance

- **Cross-Encoder**: Optimized for medical concept relevance scoring
- **NER System**: High accuracy for medical entity recognition
- **FOL Reasoner**: Efficient logical inference for medical relationships

### Scalability

- Batch processing support for large datasets
- GPU acceleration for cross-encoder models
- Efficient caching for UMLS API responses

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Set up development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NIH UMLS** - For providing comprehensive medical terminology services
- **Hugging Face** - For transformer models and tools
- **MedCPT** - For medical cross-encoder models
- **Medical Research Community** - For ongoing research and development

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Contact the development team
- Check the documentation for common solutions

## 🔗 Related Projects

- [UMLS Terminology Services](https://uts.nlm.nih.gov/)
- [MedCPT Models](https://huggingface.co/ncbi/MedCPT-Cross-Encoder)
- [Medical NLP Resources](https://github.com/allenai/scispacy)

---

**Note**: This system is designed for research and educational purposes. For clinical use, please ensure compliance with relevant medical regulations and standards. 