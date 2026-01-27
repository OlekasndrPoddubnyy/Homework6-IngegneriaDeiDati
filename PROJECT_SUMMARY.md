# Progetto Completato - Integrazione Dati Automobili e Record Linkage

## Struttura Progetto Creata

Il progetto è stato configurato con la seguente struttura completa:

### File di Configurazione
- README.md - Documentazione principale del progetto
- config.yaml - Configurazione completa (schema, blocking, modelli)
- requirements.txt - Dipendenze Python
- .gitignore - Configurazione Git
- GETTING_STARTED.md - Guida rapida all'uso
- SETUP_INSTRUCTIONS.md - Istruzioni dettagliate di setup

### Moduli Python (src/)
- __init__.py - Package initialization
- data_analysis.py - Analisi qualità dati (valori nulli, unici, statistiche)
- schema_mediation.py - Schema mediato e normalizzazioni
- ground_truth.py - Generazione ground truth con VIN
- blocking.py - Strategie B1 e B2, valutazione blocking
- record_linkage.py - Classificatore rule-based con Python Record Linkage
- dedupe_model.py - Wrapper per Dedupe, training e predizione
- ditto_model.py - Wrapper per Ditto (Deep Learning)
- evaluation.py - Valutazione completa pipeline, metriche, visualizzazioni

### Notebook (da creare in notebooks/)
Template per:
- `01_data_exploration.ipynb` - Analisi esplorativa
- `02_schema_mediation.ipynb` - Allineamento schema
- `03_ground_truth_generation.ipynb` - Generazione GT
- `04_blocking_strategies.ipynb` - Test blocking
- `05_record_linkage_rules.ipynb` - RecordLinkage
- `06_dedupe_model.ipynb` - Training Dedupe
- `07_ditto_model.ipynb` - Training Ditto
- `08_evaluation.ipynb` - Valutazione finale

### Documentazione (docs/)
- report.md - Template relazione (10 pagine) con sezioni complete
- presentation.md - Template presentazione (20 minuti, 20 slide)

### Script di Esecuzione
- run_pipeline.py - Script completo per eseguire tutte le fasi

---

## Prossimi Passi

### 1. Setup Iniziale (15 minuti)
```powershell
# Installa dipendenze
pip install -r requirements.txt

# Crea directory
mkdir data\raw\craigslist data\raw\usedcars data\processed data\ground_truth
mkdir models\dedupe models\ditto results\metrics results\visualizations notebooks
```

### 2. Download Dataset (30 minuti)
```powershell
# Setup Kaggle API: https://www.kaggle.com/docs/api
kaggle datasets download -d austinreese/craigslist-carstrucks-data -p data\raw\craigslist --unzip
kaggle datasets download -d ananaymital/us-used-cars-dataset -p data\raw\usedcars --unzip
```

### 3. Esecuzione Pipeline (6-12 ore)
Opzione A - Script automatico:
```powershell
python run_pipeline.py
```

Opzione B - Notebook interattivi:
```powershell
jupyter notebook
# Eseguire notebook 01-08 in ordine
```

### 4. Completamento Report e Presentazione (2-4 ore)
- Inserire risultati numerici in `docs/report.md`
- Aggiungere grafici/tabelle
- Completare `docs/presentation.md`
- Preparare slide PowerPoint/Google Slides

---

## Componenti Implementati

### Fase 1: Analisi Sorgenti
**Modulo**: `data_analysis.py`

**Funzionalità**:
- Analisi valori nulli per attributo (% e count)
- Analisi valori unici per attributo (cardinalità)
- Statistiche riassuntive (righe, colonne, memoria)
- Analisi specifica VIN
- Visualizzazioni (grafici a barre)
- Report testuale completo
- Confronto tra sorgenti

**Output**:
- Tabelle statistiche
- Grafici null/unique values
- Report qualità dati

---

### Fase 2: Schema Mediato
**Modulo**: `schema_mediation.py`

**Funzionalità**:
- Schema mediato con 16 attributi comuni
- Mapping Craigslist → Schema mediato
- Mapping US Used Cars → Schema mediato
- Normalizzazioni:
  - Manufacturer (chevy → chevrolet)
  - Fuel type (gasoline → gas)
  - Transmission (auto → automatic)
- Conversioni type-safe (price, year, odometer)
- Pulizia VIN

**Output**:
- Dataset allineati in `data/processed/`

---

### Fase 3: Ground Truth
**Modulo**: `ground_truth.py`

**Funzionalità**:
- Pulizia VIN (lunghezza minima, normalizzazione)
- Matching automatico via VIN comune
- Generazione balanced non-matches
- Split stratificato train/val/test (70/10/20)
- Feature engineering (include attributi nelle coppie)
- Rimozione VIN post-generazione
- Quality verification

**Output**:
- `train.csv` (70%)
- `validation.csv` (10%)
- `test.csv` (20%)
- `ground_truth_full.csv`

---

### Fase 4: Blocking
**Modulo**: `blocking.py`

**Strategie**:

**B1 - Standard Blocking**:
- Keys: `year` + `manufacturer`
- Reduction ratio atteso: 95-99%
- Pair completeness atteso: 90-95%

**B2 - Extended Blocking**:
- Keys: `year` + `manufacturer` + `model_prefix(3)`
- Reduction ratio atteso: 98-99.5%
- Pair completeness atteso: 85-92%

**Evaluator**:
- Calcolo pair completeness
- Reduction ratio
- True pairs found/missed
- Confronto strategie

**Output**:
- Candidate pairs (MultiIndex)
- Metriche blocking

---

### Fase 5: Record Linkage (Rules)
**Modulo**: `record_linkage.py`

**Metodo**: Weighted similarity scores

**Comparison functions**:
- Exact match: `year`, `fuel`, `transmission`
- String similarity: `manufacturer`, `model`, `type` (Jaro-Winkler)
- Numeric similarity: `price`, `odometer` (Gaussian)

**Score formula**:
```
Score = Σ (similarity_i × weight_i)
Match if Score ≥ 0.7
```

**Output**:
- Predictions con score
- Match details

---

### Fase 6: Dedupe
**Modulo**: `dedupe_model.py`

**Caratteristiche**:
- Supervised learning con ground truth
- Multiple field types (Exact, String, Price, Categorical)
- Active learning support
- Parallel processing (multi-core)
- Model persistence (save/load)

**Pipeline**:
1. Prepare data (dict format)
2. Train on ground truth examples
3. Predict on candidate pairs
4. Measure training/inference time

**Output**:
- Trained model in `models/dedupe/`
- Predictions con confidence scores
- Timing metrics

---

### Fase 7: Ditto
**Modulo**: `ditto_model.py`

**Caratteristiche**:
- Deep Learning (Transformer-based)
- Repository: FAIR-DA4ER
- Serialization: Record → Text
- Fine-tuning su ground truth

**Config**:
- Max length: 256 tokens
- Batch size: 16
- Epochs: 20
- Learning rate: 3e-5

**Note**: Richiede setup manuale repository + training GPU-intensive

**Output**:
- Model checkpoints
- Predictions
- Timing metrics

---

### Fase 8: Evaluation
**Modulo**: `evaluation.py`

**Metriche**:
- **Accuracy**: Precision, Recall, F1-Score, Accuracy
- **Efficiency**: Training time, Inference time
- **Confusion Matrix**: TP, FP, FN, TN

**Visualizzazioni**:
- Bar chart: Metrics comparison (P/R/F1)
- Bar chart: Timing comparison
- Heatmap: Confusion matrix per pipeline

**Report**:
- Tabella comparativa 6 pipeline
- Analisi risultati
- Recommendations

**Output**:
- `results/evaluation_report.txt`
- `results/visualizations/*.png`
- `results/metrics/*.json`

---

## Configurazione (config.yaml)

Il file `config.yaml` contiene:

### Data Paths
- Directory raw/processed/ground_truth
- Paths sorgenti Craigslist e US Used Cars

### Mediated Schema
- Lista 16 attributi comuni

### Ground Truth
- Split ratios (70/10/20)
- VIN min length (11)
- Random seed (42)

### Blocking B1/B2
- Keys per blocking
- Descrizioni strategie

### RecordLinkage
- Comparison fields con metodi e pesi
- Threshold decisionale (0.7)

### Dedupe
- Num cores (4)
- Recall weight (2)

### Ditto
- Model config (max_len, batch_size, epochs, lr)
- GitHub repo URL

### Evaluation
- Pipeline definitions (6 combinazioni)
- Metrics list
- Output directories

---

## Output Attesi

### Metriche Attese (Baseline)

| Pipeline | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| B1-RecordLinkage | 0.75-0.85 | 0.70-0.80 | 0.72-0.82 |
| B2-RecordLinkage | 0.78-0.88 | 0.65-0.75 | 0.71-0.81 |
| B1-Dedupe | 0.80-0.90 | 0.75-0.85 | 0.77-0.87 |
| B2-Dedupe | 0.82-0.92 | 0.70-0.80 | 0.75-0.85 |
| B1-Ditto | 0.85-0.95 | 0.80-0.90 | 0.82-0.92 |
| B2-Ditto | 0.87-0.97 | 0.75-0.85 | 0.80-0.90 |

*Note: Valori indicativi, dipendono da qualità dati e tuning*

---

## Consegne Finali

### 1. Relazione (10 pagine)
**File**: `docs/report.md`

**Sezioni**:
1. Introduzione e obiettivi (completato)
2. Analisi sorgenti (con tabelle/grafici) - da completare
3. Schema mediato (completato)
4. Ground truth generation (completato)
5. Strategie blocking - da completare
6. Metodi record linkage (completato)
7. Valutazione sperimentale - da completare
8. Conclusioni (completato)
9. Bibliografia - da aggiungere

### 2. Presentazione (20 minuti)
**File**: `docs/presentation.md`

**Slide**: 20 slide complete con:
- Architettura sistema (completato)
- Analisi dati (aggiungere grafici)
- Ground truth (completato)
- Blocking (completato)
- Record linkage (completato)
- Risultati (aggiungere metriche reali)
- Conclusioni (completato)

---

## Punti di Forza del Progetto

1. **Architettura Modulare**: Ogni fase è un modulo indipendente e riusabile
2. **Configurazione Centralizzata**: `config.yaml` per tuning facile
3. **Documentazione Completa**: README, guide, commenti nel codice
4. **Valutazione Rigorosa**: 6 pipeline, multiple metriche, visualizzazioni
5. **Flessibilità**: Facile testare nuove strategie/parametri
6. **Reproducibilità**: Random seeds, save/load models, versioning

---

## Note Importanti

### Ditto Setup
Ditto richiede setup manuale:
1. Clone repository: `git clone https://github.com/MarcoNapoleone/FAIR-DA4ER`
2. Installare dipendenze specifiche
3. Verificare script training/inference
4. Adattare comandi in `ditto_model.py`

### Performance
- Dataset grandi richiedono molto tempo
- Consigliato: iniziare con samples (10K-100K records)
- GPU accelera Ditto significativamente
- Dedupe può richiedere molta memoria

### Qualità Ground Truth
- Dipende criticamente da qualità VIN
- Verificare manualmente sample di coppie
- Considerare Label Studio per curating

---

## Risorse e Riferimenti

### Librerie Utilizzate
- [Python Record Linkage](https://recordlinkage.readthedocs.io/)
- [Dedupe](https://docs.dedupe.io/)
- [Ditto Paper](https://arxiv.org/abs/2004.00584)

### Dataset
- [Craigslist Cars](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- [US Used Cars](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset)

### Tools Aggiuntivi
- [Label Studio](https://labelstud.io/) - Per annotazione manuale

---

## Supporto

Per domande o problemi:
1. Consultare `GETTING_STARTED.md`
2. Verificare `SETUP_INSTRUCTIONS.md`
3. Leggere commenti nei moduli Python
4. Controllare esempi in `run_pipeline.py`

---

## Checklist Finale

### Setup
- [ ] Dipendenze installate
- [ ] Dataset scaricati
- [ ] Directory create
- [ ] Moduli importabili

### Esecuzione
- [ ] Analisi esplorativa completata
- [ ] Schema mediato validato
- [ ] Ground truth generata
- [ ] Blocking testato
- [ ] RecordLinkage eseguito (B1 e B2)
- [ ] Dedupe addestrato (B1 e B2)
- [ ] Ditto addestrato (B1 e B2)
- [ ] Valutazione completata

### Documentazione
- [ ] Metriche inserite in report.md
- [ ] Grafici aggiunti
- [ ] Presentazione completata
- [ ] Codice commentato
- [ ] README aggiornato

---

Il progetto è pronto per l'esecuzione e la valutazione.
