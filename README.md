# RahatAI - Crisis Response NLP System

Multilingual NLP system for disaster response (English, Urdu, Roman-Urdu).

## What It Does

- **Classifies** crisis messages (5 models)
- **Answers questions** using RAG
- **Extracts information** (NER)
- **Summarizes** posts
- **Detects misinformation**

## Quick Start

```bash
# Train all models
python main.py train

# Or train step-by-step
python RunScripts/STEP5_train_model3_lstm_tuned.py
python RunScripts/STEP6_train_model4_cnn.py
python RunScripts/STEP7_train_model5_transformer.py

# Other tasks
python main.py rag_setup
python main.py rag_eval
python main.py ner
python main.py summarize
python main.py misinformation
```

## Project Structure

- `Scripts/` - Main code (models, evaluation, preprocessing)
- `Data/` - Datasets (raw and processed)
- `Models/` - Saved trained models
- `Outputs/` - Results, metrics, plots
- `RunScripts/` - Step-by-step training scripts
- `Docs/` - Documentation and guides
- `config/` - Configuration file

## Models (5 Total)

1. Naive Bayes (ML) - Done
2. SVM (ML) - Done
3. LSTM (Deep Learning) - Training
4. CNN (Deep Learning) - Pending
5. XLM-RoBERTa (Transformer) - Pending

## Documentation

- `Docs/PROJECT_OVERVIEW.md` - Complete project explanation
- `Docs/TASK_CHECKLIST.md` - What to do next
- `Docs/PRESENTATION_GUIDE.md` - How to present
- `Docs/QUICK_SUMMARY.md` - Quick reference

## Requirements

```bash
pip install -r requirements.txt
```

## Status

**Completed:** Data prep, 2 models trained, evaluation system  
**In Progress:** LSTM training  
**Remaining:** 3 models, RAG, NER, Summarization, Misinformation detection

