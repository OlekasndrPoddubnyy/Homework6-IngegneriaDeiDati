# Automobile Data Integration and Record Linkage Project

## Obiettivo
Il progetto ha come obiettivo l'integrazione di dati su automobili provenienti da diverse sorgenti tramite tecniche avanzate di record linkage e entity matching.

## Sorgenti Dati
1. [Craigslist Cars & Trucks Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
2. [US Used Cars Dataset](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset)

## Struttura del Progetto

```
├── data/                           # Directory per i dati
│   ├── raw/                        # Dati originali
│   │   ├── craigslist/
│   │   └── usedcars/
│   ├── processed/                  # Dati processati
│   │   ├── craigslist_aligned.csv
│   │   └── usedcars_aligned.csv
│   └── ground_truth/               # Ground truth per valutazione
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── notebooks/                      # Jupyter notebooks per analisi
│   ├── 01_data_exploration.ipynb
│   ├── 02_schema_mediation.ipynb
│   ├── 03_ground_truth_generation.ipynb
│   ├── 04_blocking_strategies.ipynb
│   ├── 05_record_linkage_rules.ipynb
│   ├── 06_dedupe_model.ipynb
│   ├── 07_ditto_model.ipynb
│   └── 08_evaluation.ipynb
├── src/                            # Codice sorgente
│   ├── __init__.py
│   ├── data_analysis.py            # Analisi valori nulli e unici
│   ├── schema_mediation.py         # Schema mediato
│   ├── alignment.py                # Allineamento sorgenti
│   ├── ground_truth.py             # Generazione ground truth
│   ├── blocking.py                 # Strategie di blocking
│   ├── record_linkage.py           # Regole record linkage
│   ├── dedupe_model.py             # Modello Dedupe
│   ├── ditto_model.py              # Modello Ditto
│   └── evaluation.py               # Valutazione prestazioni
├── models/                         # Modelli addestrati
│   ├── dedupe/
│   └── ditto/
├── results/                        # Risultati valutazioni
│   ├── metrics/
│   └── visualizations/
├── docs/                           # Documentazione
│   ├── report.md                   # Relazione (10 pagine)
│   └── presentation.md             # Presentazione (20 minuti)
├── requirements.txt                # Dipendenze Python
├── config.yaml                     # Configurazione progetto
└── README.md                       # Questo file
```

## Fasi del Progetto

### Fase 1: Analisi delle Sorgenti
- Analizzare percentuale valori nulli per ogni attributo
- Analizzare percentuale valori unici per ogni attributo
- Caratterizzare le sorgenti dati

### Fase 2: Schema Mediato
- Definire schema mediato comune
- Allineare le sorgenti allo schema mediato

### Fase 3: Ground Truth (usando VIN)
- Generare ground truth sfruttando l'attributo VIN
- Pulire i dati rumorosi
- Creare training, validation, test set

### Fase 4: Record Linkage
- **4.D**: Definire due strategie di blocking (B1 e B2)
- **4.E**: Implementare regole con Python Record Linkage
- **4.F**: Addestrare modello con Dedupe
- **4.G**: Addestrare modello con Ditto
- **4.H**: Valutare tutte le pipeline

### Fase 5: Valutazione
Valutare 6 pipeline:
1. B1 + RecordLinkage
2. B2 + RecordLinkage
3. B1 + Dedupe
4. B2 + Dedupe
5. B1 + Ditto
6. B2 + Ditto

Metriche:
- Precision
- Recall
- F1-measure
- Tempi di training
- Tempi di inferenza

## Installazione

```bash
pip install -r requirements.txt
```

## Uso

1. Scaricare i dataset da Kaggle in `data/raw/`
2. Eseguire i notebook in ordine
3. Valutare i risultati in `results/`

## Note
- VIN viene usato solo per ground truth, poi rimosso
- Attenzione ai dati rumorosi
- Valutare uso di Label Studio per ground truth curata manualmente
