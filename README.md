# Antigravity Ranker for Redrob Hackathon

This repository contains the candidate ranking pipeline for the India Runs Data & AI Challenge.

## Requirements
- Python 3.11+
- CPU with 16GB RAM

## Setup
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Ranker
To produce the submission CSV from the candidates dataset, run:
```bash
python ranker.py --candidates ./candidates.jsonl --out ./team_PJ2001_IND.csv
```

This will run the 4-stage pipeline entirely on CPU in under 5 minutes and output the `team_PJ2001_IND.csv` file.

## Validation
You can validate the output using the official validator:
```bash
python validate_submission.py team_PJ2001_IND.csv
```
