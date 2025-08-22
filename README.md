# Setup & Run

## Prerequisites

* **Python 3.8+**
* **pip** (Python package manager)

## Installation

1. **Clone or download this project**

   ```bash
   git clone https://github.com/GathuaKiragu/humaein-case-study-1
   cd humaein-case-study-1
   ```

2. **Set up a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

### Option 1: Command Line

```bash
python3 pipeline.py
```

This will:

* Process the sample input files
* Create `resubmission_candidates.json` (eligible claims)
* Create `rejected_records.log` (problematic records)
* Print metrics in the terminal

---

### Option 2: Web API

Start the FastAPI server:

```bash
uvicorn api:app --reload
```

Then visit `http://localhost:8000` in your browser, or run:

```bash
curl -X POST http://localhost:8000/process-claims/ \
  -F "alpha_file=@emr_alpha.csv" \
  -F "beta_file=@emr_beta.json"
```

