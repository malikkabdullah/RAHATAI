# RahatAI - Complete Project Overview

## What is RahatAI?

**RahatAI** is a multilingual crisis response system that helps during disasters by:
- Understanding crisis messages in English, Urdu, and Roman-Urdu
- Classifying what type of help is needed
- Extracting important information (locations, phone numbers, resources)
- Answering questions about crisis situations
- Detecting false information

---

## Project Structure (Simple Explanation)

### 📁 Main Folders

1. **`Scripts/`** - The brain of the project
   - Contains all the code that makes everything work
   - Organized by task: classification, RAG, NER, etc.

2. **`Data/`** - All the data
   - Raw datasets from CrisisNLP and Kaggle
   - Processed/cleaned data ready for training

3. **`Models/`** - Trained AI models
   - Saved models after training (Naive Bayes, SVM, LSTM, etc.)

4. **`Outputs/`** - Results and visualizations
   - Metrics (accuracy, F1-score, etc.)
   - Plots (confusion matrices, training curves)

5. **`RunScripts/`** - Step-by-step execution
   - Simple scripts to run each step one by one
   - Easy to understand and follow

6. **`config/`** - Settings
   - Configuration file with all parameters

---

## What Does This Project Do? (6 Main Tasks)

### 1. **Text Classification** (In Progress)
**Goal:** Classify crisis messages into categories

**What it does:**
- Reads crisis posts (e.g., "Need food in Karachi")
- Classifies them into: Help Request, Offer, Report, Rumor, etc.

**Models Used:**
- **Naive Bayes** (Simple ML) Done
- **SVM** (Simple ML) Done
- **LSTM** (Deep Learning) Training
- **CNN** (Deep Learning) Waiting
- **XLM-RoBERTa** (Transformer) Waiting

**Output:** Accuracy, confusion matrix, predictions

---

### 2. **RAG (Retrieval-Augmented Generation)** (Pending)
**Goal:** Answer questions using crisis documents

**What it does:**
- Takes a question: "What resources are available in Lahore?"
- Searches through crisis documents
- Generates an answer with sources

**Components:**
- Vector store (FAISS) - stores document embeddings
- Embedding model - converts text to numbers
- LLM - generates answers

**Output:** Answers with citations, comparison with baseline

---

### 3. **NER (Named Entity Recognition)** (Pending)
**Goal:** Extract important information from text

**What it extracts:**
- **Locations:** "Karachi", "Lahore", "Islamabad"
- **Phone numbers:** "0300-1234567"
- **Resources:** "food", "medicine", "shelter"
- **People/Organizations:** Names of people or NGOs

**Languages:** English, Urdu, Roman-Urdu

**Output:** Extracted entities in structured format

---

### 4. **Summarization** (Pending)
**Goal:** Create short summaries of crisis posts

**What it does:**
- Takes many crisis posts
- Groups similar ones
- Creates a summary of each group

**Example:**
- Input: 100 posts about food shortage
- Output: "Summary: Multiple requests for food assistance in Karachi area..."

**Output:** Summaries by category/region

---

### 5. **Misinformation Detection** (Pending)
**Goal:** Detect false or misleading information

**What it does:**
- Reads a crisis post
- Determines if it's true or false
- Binary classification: True/False

**Output:** Misinformation predictions with confidence scores

---

### 6. **Evaluation** (Partial)
**Goal:** Measure how well models perform

**Metrics Used:**
- Accuracy: How many correct predictions?
- Precision: How many of predicted positives are actually positive?
- Recall: How many actual positives were found?
- F1-Score: Balance of precision and recall
- AUC: Overall model performance
- Confusion Matrix: Visual representation of errors

**Output:** JSON files with all metrics, plots

---

## How Models Work (Simple Explanation)

### Machine Learning Models (Naive Bayes, SVM)
- **How:** Learn patterns from examples
- **Input:** Text → Convert to numbers (TF-IDF)
- **Process:** Find patterns in numbers
- **Output:** Classification label
- **Speed:** Very fast
- **Accuracy:** Good for simple tasks

### Deep Learning Models (LSTM, CNN)
- **How:** Neural networks that learn complex patterns
- **Input:** Text → Convert to sequences → Embeddings
- **Process:** Multiple layers process information
- **Output:** Classification label
- **Speed:** Slower (needs GPU for best performance)
- **Accuracy:** Better for complex patterns

### Transformer Models (XLM-RoBERTa)
- **How:** Advanced neural network, understands context
- **Input:** Text → Tokenization → Embeddings
- **Process:** Attention mechanism understands relationships
- **Output:** Classification label
- **Speed:** Slowest (needs GPU)
- **Accuracy:** Best for multilingual tasks

---

## Data Flow (Step by Step)

```
1. Raw Data (CrisisNLP, Kaggle)
   ↓
2. Data Preprocessing (clean text, encode labels)
   ↓
3. Train Models (5 different models)
   ↓
4. Evaluate Models (calculate metrics)
   ↓
5. Save Results (models, metrics, plots)
   ↓
6. Use Models (classification, NER, etc.)
```

---

## Current Status

### Completed
- Data preparation and preprocessing
- Naive Bayes model training
- SVM model training
- Evaluation metrics system
- Project structure setup

### In Progress
- LSTM model training (running in background)

### Pending
- CNN model training
- Transformer model training
- RAG pipeline setup
- NER implementation
- Summarization implementation
- Misinformation detection
- QA dataset creation (100 pairs)

---

## Key Files Explained

### `main.py`
- **Purpose:** Main entry point to run all tasks
- **Usage:** `python main.py [task]`
- **Tasks:** train, rag_setup, rag_eval, ner, summarize, misinformation

### `config/config.yaml`
- **Purpose:** All settings in one place
- **Contains:** Model parameters, paths, evaluation settings

### `Scripts/classification/`
- **Purpose:** All classification models
- **Files:**
  - `ml_models.py` - Naive Bayes, SVM
  - `dl_models.py` - LSTM, CNN
  - `transformer_models.py` - XLM-RoBERTa

### `Scripts/evaluation/metrics.py`
- **Purpose:** Calculate all evaluation metrics
- **Functions:** Accuracy, Precision, Recall, F1, AUC, etc.

### `RunScripts/STEP*.py`
- **Purpose:** Step-by-step scripts for training
- **Why:** Easy to run one step at a time, understand what's happening

---

## Technologies Used

- **Python 3.x** - Programming language
- **TensorFlow/Keras** - Deep learning framework
- **Scikit-learn** - Machine learning library
- **HuggingFace Transformers** - Pre-trained models
- **FAISS** - Vector database for RAG
- **LangChain** - RAG framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Plotting

---

## Evaluation Requirements (What You Need to Show)

### For Classification Models:
1. **At least 2 ML models** (Naive Bayes, SVM)
2. **At least 2 DL models** (LSTM, CNN)
3. **At least 1 Transformer model** (XLM-RoBERTa)
4. **Different embedding techniques** (TF-IDF, Word Embeddings, Transformer embeddings)
5. **All metrics:** Accuracy, Precision, Recall, F1, AUC, EM, Top-k, Confusion Matrix
6. **Training/Validation plots:** Accuracy and Loss curves

### For RAG:
1. **Custom QA dataset:** 100 question-answer pairs with citations
2. **RAG vs Baseline comparison:** Factuality, Completeness, Faithfulness, Safety
3. **Human and LLM judgment:** Both evaluation methods

---

## Next Steps (What to Do)

1. **Wait for LSTM to finish** (currently training)
2. **Train CNN** (run `STEP6_train_model4_cnn.py`)
3. **Train Transformer** (run `STEP7_train_model5_transformer.py`)
4. **Set up RAG** (create vector store, embeddings)
5. **Create QA dataset** (100 pairs)
6. **Implement NER** (extract entities)
7. **Implement Summarization** (generate summaries)
8. **Implement Misinformation Detection** (detect false info)
9. **Final evaluation** (compare all models, generate report)

---

## How to Present This Project

### Option 1: Simple Presentation (Recommended)
Create a PowerPoint/PDF with:
1. **Title Slide:** RahatAI - Crisis Response NLP System
2. **Problem Statement:** Why this is needed
3. **Solution Overview:** What the system does
4. **Architecture:** Simple diagram of components
5. **Models:** Show all 5 models with results
6. **Results:** Metrics, confusion matrices, plots
7. **Demo:** Show classification, NER, RAG in action
8. **Conclusion:** Summary and future work

### Option 2: Interactive Demo
- Create a simple web interface (Flask/Streamlit)
- Allow users to input text
- Show classification, NER extraction, RAG answers
- Display results visually

### Option 3: Jupyter Notebook
- Create a comprehensive notebook
- Show step-by-step execution
- Include visualizations
- Explain each component

---

## Tips for Presentation

1. **Start Simple:** Explain the problem first
2. **Show Results:** Visuals (plots, confusion matrices) are powerful
3. **Compare Models:** Show which model performs best
4. **Real Examples:** Use actual crisis posts as examples
5. **Multilingual:** Show it works in English, Urdu, Roman-Urdu
6. **Practical Use:** Explain how this helps in real disasters

---

## Common Questions & Answers

**Q: Why 5 different models?**
A: To compare different approaches and show which works best for crisis response.

**Q: What makes this multilingual?**
A: Uses XLM-RoBERTa which understands multiple languages, and preprocessing handles Urdu/Roman-Urdu.

**Q: How is this different from ChatGPT?**
A: This is specialized for crisis response, uses domain-specific data, and includes RAG for accurate, cited answers.

**Q: What's the most important part?**
A: Classification (identifying what type of help is needed) is the core, everything else builds on it.

---

## Summary

**RahatAI** is a complete NLP system for crisis response that:
- Classifies crisis messages (5 models)
- Extracts important information (NER)
- Answers questions (RAG)
- Summarizes information
- Detects misinformation

**Status:** 2/5 models trained, 3 more to go, then other modules.

**Goal:** Help responders understand and act on crisis information faster and more accurately.

