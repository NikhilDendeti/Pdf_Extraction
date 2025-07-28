# Adobe Hack - RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline to solve Adobe Hack challenges using a combination of PDFs and structured input JSONs. The core logic processes inputs and references the relevant documents to generate answers using an LLM.

---

## 🗂️ Project Structure

```
Adobe_Hack/
│
├── 1b/                         # Challenge input/output folder
│   ├── Collection 1/
│   ├── Collection 2/
│   └── Collection 3/
│       ├── challenge1b_input.json
│       ├── challenge1b_output.json
│       └── PDFs/
│           └── All reference documents (PDFs)
│
├── ragpipeline/               # Main pipeline code
│   ├── main.py                # Entry point to run the pipeline
│   ├── utils.py               # Helper functions for file I/O, PDF parsing, etc.
│   ├── config.py              # Configuration file (e.g., LLM settings)
│   └── test.py                # Test script to validate outputs
│
├── venv/                      # Virtual environment
│
├── .gitignore                 # Files to ignore in git
├── requirements.txt           # Python dependencies
```

---

## ▶️ How to Run

### 📌 Prerequisites

1. Python 3.8+
2. Virtual environment activated (recommended)
3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

### 🚀 Run the pipeline

To run the pipeline on **Collection 3** under `1b`, use:

```bash
python3 ragpipeline/main.py "../1b/Collection 3/challenge1b_input.json" "../1b/Collection 3/PDFs"
```

Replace the input path and PDFs folder with those of other collections as needed.

---

## 📦 Output

The script will generate responses for each question in the input JSON, using the relevant content from the PDF documents, and write the output in a file like:

```
../1b/Collection 3/challenge1b_output.json
```

---

## 📁 Example Collections

* **Collection 1**: Travel Planning
* **Collection 2**: Management Creation
* **Collection 3**: Menu Planning

Each contains:

* An input JSON (`challenge1b_input.json`)
* A set of PDFs for reference
* An output JSON (`challenge1b_output.json`)

---

## 📜 License

MIT License (if applicable)

---

## 🤝 Contributing

Feel free to raise issues or PRs for improvements or additional features.
