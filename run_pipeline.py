"""
Run the whole project end to end:

    python run_pipeline.py

1. generate raw data   ->  data/raw/tickets_raw.csv
2. clean & validate    ->  data/processed/tickets_clean.csv, reports/data_quality_report.md
3. SQL KPI analysis    ->  reports/sql_results.md
4. EDA charts          ->  reports/figures/eda_*.png
5. train AI models     ->  models/*.joblib, reports/metrics.json, reports/figures/*
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

STEPS = [
    ("Generate raw data", ROOT / "data" / "generate_tickets.py"),
    ("Clean & validate", ROOT / "src" / "clean.py"),
    ("SQL KPI analysis", ROOT / "src" / "sql_analysis.py"),
    ("EDA charts", ROOT / "src" / "eda.py"),
    ("Train AI models", ROOT / "src" / "train.py"),
]

for i, (name, path) in enumerate(STEPS, 1):
    print(f"\n{'=' * 70}\nStep {i}/{len(STEPS)}: {name}\n{'=' * 70}")
    runpy.run_path(str(path), run_name="__main__")

print("\nDone. Launch the app with:  streamlit run app.py")
