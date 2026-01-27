# Guida Rapida al Progetto

## Getting Started

### 1. Installazione Dipendenze

```bash
cd "c:\Users\O.Poddubnyy\Downloads\Homeworks\Homework 6"
pip install -r requirements.txt
```

### 2. Scarica i Dataset

I dataset vanno scaricati da Kaggle e posizionati in `data/raw/`:

```bash
# Installa Kaggle CLI
pip install kaggle

# Configura le credenziali Kaggle (segui: https://www.kaggle.com/docs/api)

# Scarica Craigslist dataset
kaggle datasets download -d austinreese/craigslist-carstrucks-data -p data/raw/craigslist --unzip

# Scarica US Used Cars dataset
kaggle datasets download -d ananaymital/us-used-cars-dataset -p data/raw/usedcars --unzip
```

### 3. Struttura Directory

Crea le directory necessarie:

```bash
mkdir -p data/raw/craigslist
mkdir -p data/raw/usedcars
mkdir -p data/processed
mkdir -p data/ground_truth
mkdir -p models/dedupe
mkdir -p models/ditto
mkdir -p results/metrics
mkdir -p results/visualizations
```

## Workflow del Progetto

### Fase 1: Analisi Esplorativa
**Notebook**: `notebooks/01_data_exploration.ipynb`

```python
# Carica e analizza le sorgenti
from src.data_analysis import DataAnalyzer

analyzer1 = DataAnalyzer(df_craigslist, 'Craigslist')
analyzer1.analyze_null_values()
analyzer1.analyze_unique_values()
analyzer1.plot_null_values()
```

**Output**:
- Statistiche valori nulli
- Statistiche valori unici
- Report qualità dati

### Fase 2: Schema Mediato
**Notebook**: `notebooks/02_schema_mediation.ipynb`

```python
from src.schema_mediation import MediatedSchema, SourceAligner

schema = MediatedSchema('config.yaml')
aligner = SourceAligner(schema)

df1_aligned = aligner.align_craigslist(df_craigslist)
df2_aligned = aligner.align_usedcars(df_usedcars)
```

**Output**:
- `data/processed/craigslist_aligned.csv`
- `data/processed/usedcars_aligned.csv`

### Fase 3: Ground Truth
**Notebook**: `notebooks/03_ground_truth_generation.ipynb`

```python
from src.ground_truth import GroundTruthGenerator

gt_gen = GroundTruthGenerator(df1_aligned, df2_aligned)
gt_gen.find_matches()
gt_gen.generate_non_matches()
ground_truth = gt_gen.create_ground_truth()

gt_gen.save_ground_truth(ground_truth, 'data/ground_truth', remove_vin=True)
```

**Output**:
- `data/ground_truth/train.csv`
- `data/ground_truth/validation.csv`
- `data/ground_truth/test.csv`

### Fase 4: Blocking Strategies
**Notebook**: `notebooks/04_blocking_strategies.ipynb`

```python
from src.blocking import StandardBlocking, ExtendedBlocking

b1 = StandardBlocking()
pairs_b1 = b1.generate_pairs(df1, df2)

b2 = ExtendedBlocking()
pairs_b2 = b2.generate_pairs(df1, df2)
```

**Output**:
- Statistiche blocking
- Confronto strategie

### Fase 5: Record Linkage (Rules)
**Notebook**: `notebooks/05_record_linkage_rules.ipynb`

```python
from src.record_linkage import RecordLinkageClassifier

# B1 + RecordLinkage
classifier = RecordLinkageClassifier()
results_b1_rl = classifier.classify(df1, df2, pairs_b1)

# B2 + RecordLinkage
results_b2_rl = classifier.classify(df1, df2, pairs_b2)
```

### Fase 6: Dedupe
**Notebook**: `notebooks/06_dedupe_model.ipynb`

```python
from src.dedupe_model import DedupeModel

model = DedupeModel()

# B1 + Dedupe
training_time = model.train(df1, df2, ground_truth_train)
predictions_b1_dedupe, inference_time = model.predict(df1, df2, pairs_b1)

# B2 + Dedupe
predictions_b2_dedupe, inference_time = model.predict(df1, df2, pairs_b2)
```

### Fase 7: Ditto
**Notebook**: `notebooks/07_ditto_model.ipynb`

```python
from src.ditto_model import DittoModel

model = DittoModel()
model.setup_ditto()

# Prepara dati
model.prepare_data_for_ditto(ground_truth, df1, df2, 'data/ditto')

# Training
training_time = model.train('data/ditto/train.csv', 'data/ditto/val.csv')

# Predizione
predictions, inference_time = model.predict('data/ditto/test.csv')
```

### Fase 8: Valutazione
**Notebook**: `notebooks/08_evaluation.ipynb`

```python
from src.evaluation import PipelineEvaluator

evaluator = PipelineEvaluator(ground_truth_test)

# Valuta tutte le pipeline
evaluator.evaluate_predictions(results_b1_rl, 'B1-RecordLinkage', df1, df2)
evaluator.evaluate_predictions(results_b2_rl, 'B2-RecordLinkage', df1, df2)
evaluator.evaluate_predictions(predictions_b1_dedupe, 'B1-Dedupe', df1, df2)
evaluator.evaluate_predictions(predictions_b2_dedupe, 'B2-Dedupe', df1, df2)
evaluator.evaluate_predictions(predictions_b1_ditto, 'B1-Ditto', df1, df2)
evaluator.evaluate_predictions(predictions_b2_ditto, 'B2-Ditto', df1, df2)

# Confronto
comparison = evaluator.compare_pipelines()
evaluator.plot_metrics_comparison()
evaluator.generate_report('results/evaluation_report.txt')
```

## Troubleshooting

### Problema: Dataset troppo grande

```python
# Usa un campione per test rapidi
df_sample = df.sample(n=10000, random_state=42)
```

### Problema: Dedupe richiede troppa memoria

```python
# Riduci batch size o usa meno core
model = DedupeModel(num_cores=2)
```

### Problema: Ditto non installato

```bash
# Clona repository
git clone https://github.com/MarcoNapoleone/FAIR-DA4ER
cd FAIR-DA4ER
pip install -r requirements.txt
```

## Checklist Progetto

### Completato
- [X] Dataset scaricati
- [X] Analisi esplorativa completata
- [X] Schema mediato definito
- [X] Sorgenti allineate
- [X] Ground truth generata
- [X] Blocking B1 implementato
- [X] Blocking B2 implementato
- [ ] RecordLinkage testato
- [ ] Dedupe addestrato
- [ ] Ditto addestrato
- [ ] Tutte le 6 pipeline valutate
- [ ] Report generato
- [ ] Presentazione preparata

## Output Attesi

### File Dati
```
data/
├── processed/
│   ├── craigslist_aligned.csv
│   └── usedcars_aligned.csv
├── ground_truth/
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── ground_truth_full.csv
```

### Modelli
```
models/
├── dedupe/
│   ├── training.json
│   └── settings
├── ditto/
│   └── checkpoint-*/
```

### Risultati
```
results/
├── metrics/
│   ├── b1_recordlinkage_metrics.json
│   ├── b2_recordlinkage_metrics.json
│   ├── b1_dedupe_metrics.json
│   ├── b2_dedupe_metrics.json
│   ├── b1_ditto_metrics.json
│   └── b2_ditto_metrics.json
├── visualizations/
│   ├── metrics_comparison.png
│   ├── timing_comparison.png
│   └── confusion_matrices/
└── evaluation_report.txt
```

### Documentazione
```
docs/
├── report.md (10 pagine)
└── presentation.md (20 minuti)
```

## Suggerimenti per il Successo

1. **Inizia con un subset**: Test su 10K records prima di full dataset
2. **Monitora la memoria**: Record linkage può essere memory-intensive
3. **Salva checkpoint**: Salva risultati intermedi frequentemente
4. **Documenta scelte**: Annota decisioni e razionale nel notebook
5. **Visualizza risultati**: Grafici aiutano a identificare pattern
6. **Confronta metriche**: Non solo F1, guarda precision/recall trade-off

## Risorse Utili

- [Python Record Linkage Docs](https://recordlinkage.readthedocs.io/)
- [Dedupe Documentation](https://docs.dedupe.io/)
- [Ditto Paper](https://arxiv.org/abs/2004.00584)
- [Label Studio](https://labelstud.io/) (per ground truth curata)

## Problemi Comuni

### Import Error
```bash
# Assicurati che src/ sia nel PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

### VIN non trovato
Verifica che la colonna VIN sia presente e con il nome corretto nei dataset

### Out of Memory
Riduci la dimensione del dataset o usa blocking più aggressivo

---

Per ulteriori informazioni consultare la documentazione completa nel file PROJECT_SUMMARY.md e SETUP_INSTRUCTIONS.md.
