# Setup and Execution Instructions

## Quick Start Commands

### Windows PowerShell

```powershell
# 1. Navigate to project directory
cd "c:\Users\O.Poddubnyy\Downloads\Homeworks\Homework 6"

# 2. Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create directory structure
New-Item -ItemType Directory -Force -Path "data\raw\craigslist"
New-Item -ItemType Directory -Force -Path "data\raw\usedcars"
New-Item -ItemType Directory -Force -Path "data\processed"
New-Item -ItemType Directory -Force -Path "data\ground_truth"
New-Item -ItemType Directory -Force -Path "models\dedupe"
New-Item -ItemType Directory -Force -Path "models\ditto"
New-Item -ItemType Directory -Force -Path "results\metrics"
New-Item -ItemType Directory -Force -Path "results\visualizations"
New-Item -ItemType Directory -Force -Path "notebooks"

# 5. Download datasets (requires Kaggle API setup)
# First, setup Kaggle credentials: https://www.kaggle.com/docs/api
kaggle datasets download -d austinreese/craigslist-carstrucks-data -p "data\raw\craigslist" --unzip
kaggle datasets download -d ananaymital/us-used-cars-dataset -p "data\raw\usedcars" --unzip
```

## Execution Order

### Step 1: Data Exploration
```powershell
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Goals:**
- Load datasets
- Analyze null values
- Analyze unique values
- Generate quality reports

### Step 2: Schema Mediation
```powershell
jupyter notebook notebooks/02_schema_mediation.ipynb
```

**Goals:**
- Define mediated schema
- Align Craigslist dataset
- Align US Used Cars dataset
- Save processed data

### Step 3: Ground Truth Generation
```powershell
jupyter notebook notebooks/03_ground_truth_generation.ipynb
```

**Goals:**
- Clean VIN data
- Find matching pairs
- Generate non-matching pairs
- Split train/val/test
- Remove VIN from ground truth

### Step 4: Blocking Strategies
```powershell
jupyter notebook notebooks/04_blocking_strategies.ipynb
```

**Goals:**
- Implement B1 (Standard Blocking)
- Implement B2 (Extended Blocking)
- Compare reduction ratios
- Evaluate pair completeness

### Step 5: Record Linkage (Rules)
```powershell
jupyter notebook notebooks/05_record_linkage_rules.ipynb
```

**Goals:**
- Configure similarity functions
- Apply B1 + RecordLinkage
- Apply B2 + RecordLinkage
- Collect results

### Step 6: Dedupe Model
```powershell
jupyter notebook notebooks/06_dedupe_model.ipynb
```

**Goals:**
- Configure Dedupe fields
- Train with ground truth
- Apply B1 + Dedupe
- Apply B2 + Dedupe
- Measure training/inference time

### Step 7: Ditto Model
```powershell
jupyter notebook notebooks/07_ditto_model.ipynb
```

**Goals:**
- Setup Ditto repository
- Prepare data format
- Train model
- Apply B1 + Ditto
- Apply B2 + Ditto
- Measure training/inference time

### Step 8: Evaluation
```powershell
jupyter notebook notebooks/08_evaluation.ipynb
```

**Goals:**
- Evaluate all 6 pipelines
- Calculate metrics (P, R, F1)
- Compare timing
- Generate visualizations
- Create final report

## Testing Individual Modules

```powershell
# Test data analysis
python -c "from src.data_analysis import DataAnalyzer; print('OK')"

# Test schema mediation
python -c "from src.schema_mediation import MediatedSchema; print('OK')"

# Test ground truth generation
python -c "from src.ground_truth import GroundTruthGenerator; print('OK')"

# Test blocking
python -c "from src.blocking import StandardBlocking; print('OK')"

# Test record linkage
python -c "from src.record_linkage import RecordLinkageClassifier; print('OK')"

# Test dedupe
python -c "from src.dedupe_model import DedupeModel; print('OK')"

# Test ditto
python -c "from src.ditto_model import DittoModel; print('OK')"

# Test evaluation
python -c "from src.evaluation import PipelineEvaluator; print('OK')"
```

## Running Jupyter Lab (Alternative)

```powershell
# Start Jupyter Lab
jupyter lab

# Navigate to notebooks/ and execute in order
```

## Expected Runtime

| Phase | Estimated Time |
|-------|---------------|
| Data Exploration | 15-30 min |
| Schema Mediation | 10-20 min |
| Ground Truth | 20-40 min |
| Blocking | 10-15 min |
| RecordLinkage | 30-60 min |
| Dedupe Training | 1-3 hours |
| Ditto Training | 3-8 hours |
| Evaluation | 20-40 min |
| **TOTAL** | **~6-12 hours** |

*Note: Times depend on dataset size and hardware*

## Hardware Recommendations

### Minimum
- CPU: 4 cores
- RAM: 8 GB
- Storage: 10 GB free

### Recommended
- CPU: 8+ cores
- RAM: 16+ GB
- Storage: 20+ GB free
- GPU: Optional (speeds up Ditto)

## Troubleshooting

### Issue: "kaggle: command not found"
```powershell
pip install kaggle
# Then setup credentials as per Kaggle docs
```

### Issue: "Module not found"
```powershell
# Ensure you're in project root
$env:PYTHONPATH = "$(Get-Location);$env:PYTHONPATH"
```

### Issue: Out of Memory
```python
# In notebooks, use smaller sample
df_sample = df.sample(n=10000, random_state=42)
```

### Issue: Jupyter kernel crashes
```powershell
# Increase memory limit
$env:JUPYTER_MEMORY_LIMIT = "4096"
jupyter notebook
```

## Verification Checklist

Before starting, verify:
- [ ] Python 3.8+ installed
- [ ] pip working
- [ ] Kaggle API configured
- [ ] At least 10GB free disk space
- [ ] Internet connection for downloads

After setup, verify:
- [ ] All directories created
- [ ] Datasets downloaded
- [ ] All modules importable
- [ ] Jupyter working

## Support

If you encounter issues:
1. Check the error message carefully
2. Verify all dependencies installed
3. Check dataset paths
4. Consult GETTING_STARTED.md for detailed guidance
5. Review notebook comments and documentation

## Final Report Generation

After completing all notebooks:

```powershell
# Generate final evaluation report
python -c "
from src.evaluation import PipelineEvaluator
import pandas as pd

gt = pd.read_csv('data/ground_truth/test.csv')
evaluator = PipelineEvaluator(gt)
# Load all results and evaluate
# ...
evaluator.generate_report('results/final_report.txt')
"
```

## Documentation

Complete the following before submission:
- [ ] Fill in results in `docs/report.md`
- [ ] Add visualizations to `results/visualizations/`
- [ ] Update `docs/presentation.md` with actual metrics
- [ ] Review all notebooks for clarity
- [ ] Add comments to non-obvious code

---

Per ulteriori dettagli consultare la documentazione completa del progetto.
