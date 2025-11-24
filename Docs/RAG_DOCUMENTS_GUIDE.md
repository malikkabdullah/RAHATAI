# RAG Documents Guide

## Where to Find Documents for RAG

The RAG (Retrieval-Augmented Generation) system needs documents about disaster response, crisis management, emergency procedures, etc. to answer questions.

---

## Document Location

**Place your documents here:**
```
Data/documents/
```

**Supported formats:**
- PDF files (`.pdf`) - Recommended
- Text files (`.txt`)
- Markdown files (`.md`)

---

## Where to Get Documents

### Option 1: Government/Organization Websites (Recommended)

**International Organizations:**
- **UN OCHA (Office for the Coordination of Humanitarian Affairs)**
  - Website: https://www.unocha.org/
  - Search for: "disaster response guidelines", "emergency procedures"
  - Download PDF reports and guides

- **Red Cross / Red Crescent**
  - Website: https://www.ifrc.org/
  - Search for: "disaster response", "emergency preparedness"
  - Download manuals and guides

- **FEMA (Federal Emergency Management Agency)**
  - Website: https://www.fema.gov/
  - Search for: "disaster response", "emergency management"
  - Download PDF guides

**Pakistani Organizations:**
- **NDMA (National Disaster Management Authority)**
  - Website: https://www.ndma.gov.pk/
  - Download: Disaster management plans, emergency response guides

- **PDMA (Provincial Disaster Management Authority)**
  - Search for provincial disaster management documents

### Option 2: Academic/Research Sources

- **Research Papers:**
  - Google Scholar: Search "disaster response", "crisis management"
  - Download PDF papers about emergency response

- **University Resources:**
  - Disaster management course materials
  - Emergency response protocols

### Option 3: Create Your Own Documents

You can create simple text files with:
- Emergency contact information
- Disaster response procedures
- Resource allocation guidelines
- Evacuation procedures

**Example: `Data/documents/emergency_contacts.txt`**
```
Emergency Contacts:
- National Emergency: 112
- Fire Department: 16
- Ambulance: 115
- Police: 15

Disaster Response Centers:
- Karachi: [address]
- Lahore: [address]
- Islamabad: [address]
```

### Option 4: Use Existing Project Data

You can also convert your existing dataset into a document:

```python
# Create a document from your training data
import pandas as pd

df = pd.read_csv("Data/Preprocessed/train_preprocessed.csv")
# Combine all texts into a document
with open("Data/documents/crisis_dataset.txt", "w", encoding="utf-8") as f:
    for text in df['text']:
        f.write(text + "\n\n")
```

---

## Recommended Documents

For a complete RAG system, include:

1. **Emergency Response Guide** (PDF)
   - Procedures for different disaster types
   - Step-by-step response protocols

2. **Contact Information** (TXT)
   - Emergency numbers
   - Organization contacts
   - Resource centers

3. **Resource Allocation** (PDF/TXT)
   - How to request aid
   - Where to find resources
   - Distribution procedures

4. **Evacuation Procedures** (PDF)
   - Safe zones
   - Evacuation routes
   - Shelter locations

5. **Medical Emergency Guide** (PDF)
   - First aid procedures
   - Medical facility locations
   - Health emergency protocols

---

## How to Set Up

### Step 1: Create Documents Folder
```bash
# Already created at: Data/documents/
```

### Step 2: Add Your Documents
Place PDF/text files in `Data/documents/`

Example:
```
Data/documents/
  ├── disaster_response_guide.pdf
  ├── emergency_contacts.txt
  ├── evacuation_procedures.pdf
  └── resource_allocation.txt
```

### Step 3: Update RAG Setup Script

Edit `Scripts/rag/setup_rag.py` or create a simple script:

```python
from Scripts.rag.setup_rag import setup_rag_vectorstore
from pathlib import Path

# List your document paths
doc_paths = [
    "Data/documents/disaster_response_guide.pdf",
    "Data/documents/emergency_contacts.txt",
    # Add more paths here
]

# Setup vector store
setup_rag_vectorstore(doc_paths=doc_paths)
```

### Step 4: Run RAG Setup
```bash
python main.py rag_setup
```

Or create a custom script:
```python
# RunScripts/setup_rag_custom.py
from Scripts.rag.setup_rag import setup_rag_vectorstore

doc_paths = [
    "Data/documents/your_file1.pdf",
    "Data/documents/your_file2.txt",
]

setup_rag_vectorstore(doc_paths=doc_paths)
```

---

## Quick Start (Minimal Setup)

If you don't have documents yet, you can create a simple one:

1. Create `Data/documents/sample_guide.txt`:
```
Disaster Response Guide

Emergency Contacts:
- National Emergency: 112
- Fire: 16
- Ambulance: 115

During Floods:
1. Move to higher ground
2. Avoid walking through floodwater
3. Contact emergency services

During Earthquakes:
1. Drop, Cover, Hold
2. Stay away from windows
3. Evacuate if building is unsafe

Resources Available:
- Food distribution centers in major cities
- Medical aid at designated hospitals
- Temporary shelters at community centers
```

2. Run setup:
```bash
python -c "from Scripts.rag.setup_rag import setup_rag_vectorstore; setup_rag_vectorstore(doc_paths=['Data/documents/sample_guide.txt'])"
```

---

## Document Requirements

**Minimum:**
- 1-2 documents (even simple text files work)
- Total content: ~10-20 pages worth of text

**Recommended:**
- 3-5 documents
- Mix of PDFs and text files
- Total content: 50-100 pages

**For Best Results:**
- 5-10 comprehensive documents
- Well-structured content
- Clear sections and headings
- Total content: 100+ pages

---

## Tips

1. **Start Small**: Even 1-2 simple documents work for testing
2. **Use Existing Data**: Convert your training dataset to a document
3. **Government Sources**: Best quality, official information
4. **Organize Content**: Well-structured documents work better
5. **Multiple Formats**: Mix PDFs and text files

---

## Troubleshooting

**Problem**: "No documents found"
- **Solution**: Make sure files are in `Data/documents/` folder
- Check file paths are correct

**Problem**: "Error loading PDF"
- **Solution**: Ensure PDF is not password-protected
- Try converting to text file first

**Problem**: "Vector store empty"
- **Solution**: Check documents have actual text content
- Verify files are readable

---

## Next Steps After Adding Documents

1. **Setup Vector Store:**
   ```bash
   python main.py rag_setup
   ```

2. **Create QA Pairs:**
   - Use template: `python Scripts/rag/create_qa_dataset.py`
   - Create 100 question-answer pairs based on your documents

3. **Evaluate RAG:**
   ```bash
   python main.py rag_eval
   ```

---

*For more help, see: `Scripts/rag/setup_rag.py`*

