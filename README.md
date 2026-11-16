# Rahul zip: synthetic dairy health paper and dashboard demo

This bundle contains an IEEE journal-style manuscript, its editable Word and LaTeX sources, a PDF export, generated figures and tables, a reproducible synthetic-data script, the generated CSV, and a GitHub Pages demonstration dashboard.

The bibliography has exactly 18 unique DOI-bearing sources. The reference list and search notes are included in `paper/`.

## Open the paper

- `paper/paper.docx` is the editable two-column Word manuscript.
- `paper/paper.pdf` is the matching PDF export from Word.
- `paper/paper.tex` is the IEEEtran LaTeX source. Compile from the `paper/` directory; figure files are referenced from `../figures/`.
- `paper/references.bib` and `paper/references_apa_numbered.txt` contain the numbered APA-style references and DOI records.

The author and affiliation line is left for the author to complete. Replace it before submission.

## GitHub Pages dashboard

The root `index.html` is a self-contained public demonstration. It animates simulated herd readings in the browser and requires no build step or server. To publish from a repository root, copy the bundle contents into your repository, commit and push, then select **Settings > Pages > Deploy from a branch > main > /(root)**. GitHub Pages will display the public sample dashboard.

The Pages version is not connected to a backend. It has no authenticated farmer/admin roles, protected endpoints, real IoT sensors, or model inference. Never add credentials or private animal data to this static page.

## Run the local role-controlled demo API

Use Python 3.11 or newer, install `requirements-dashboard.txt`, then run from this folder:

```powershell
python -m pip install -r requirements-dashboard.txt
python -m uvicorn app.farm_local:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/farm`. The server prints temporary farmer and admin passwords at startup. The farmer can read the herd and alerts. The admin can also acknowledge an alert. Sessions and readings are in memory and reset on restart. Keep the server bound to loopback for this demo.

## Reproduce the model experiment

Install the packages in `requirements-ml.txt`, then run:

```powershell
python app/synthetic_cow_health_hybrid.py
```

The script uses seed 42 and prints the evaluation JSON and feature importances. The displayed results are synthetic. The synthetic target influences the sensor distributions, so the high test scores do not estimate clinical accuracy. The external Random Forest and XGBoost values in the paper come from Kaur and Virk (2025) and use a different dataset and protocol.

## Package map

- `index.html`: public GitHub Pages demo
- `paper/`: Word, LaTeX, PDF, and references
- `figures/`: source PNG figures used by both manuscript formats
- `tables/`: machine-readable result tables
- `data/`: 10,000-row CSV and exact experiment metrics
- `app/`: synthetic model script and standalone local dashboard API
- `requirements-*.txt`: optional ML or local dashboard dependencies

This is a research/software demonstration. It is not a veterinary diagnostic tool.
