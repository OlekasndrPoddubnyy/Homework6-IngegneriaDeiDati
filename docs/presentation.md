# Presentazione: Integrazione Dati Automobili e Record Linkage

## Slide 1: Titolo
**Integrazione Dati Automobili con Tecniche di Record Linkage**

*Confronto di Pipeline per Entity Matching*

[Nome Studente]
[Corso - Data]

---

## Slide 2: Agenda (0:30)

1. Introduzione e Obiettivi
2. Architettura del Sistema
3. Analisi delle Sorgenti
4. Schema Mediato e Allineamento
5. Ground Truth Generation
6. Strategie di Blocking
7. Metodi di Record Linkage
8. Valutazione Sperimentale
9. Risultati e Conclusioni

**Durata totale: 20 minuti**

---

## Slide 3: Problema e Obiettivi (1:00)

### Il Problema
- Dati su automobili distribuiti in **multiple sorgenti online**
- Schema eterogeneo, qualità variabile
- Necessità di **integrare** e **dedupplicare**

### Obiettivi
- Integrare 2 sorgenti (Craigslist + US Used Cars)
- Definire schema mediato comune
- Implementare e valutare **6 pipeline** di record linkage
- Confrontare **3 metodi** × **2 strategie di blocking**

---

## Slide 4: Architettura del Sistema (2:00)

```
┌─────────────────────────────────────────────────────┐
│              SORGENTI DATI                          │
│   Craigslist Cars    │    US Used Cars              │
└──────────────┬───────┴──────────┬───────────────────┘
               │                  │
               ▼                  ▼
       ┌───────────────────────────────┐
       │   SCHEMA MEDIATO              │
       │   - Normalizzazione           │
       │   - Pulizia dati              │
       └──────────┬────────────────────┘
                  │
                  ▼
       ┌───────────────────────────────┐
       │   GROUND TRUTH (VIN-based)    │
       │   Train / Val / Test          │
       └──────────┬────────────────────┘
                  │
                  ▼
       ┌───────────────────────────────┐
       │   BLOCKING STRATEGIES         │
       │   B1: Year + Manufacturer     │
       │   B2: Year + Manuf + Model    │
       └──────────┬────────────────────┘
                  │
                  ▼
       ┌───────────────────────────────┐
       │   RECORD LINKAGE              │
       │   - RecordLinkage (rules)     │
       │   - Dedupe (ML)               │
       │   - Ditto (Deep Learning)     │
       └──────────┬────────────────────┘
                  │
                  ▼
       ┌───────────────────────────────┐
       │   EVALUATION                  │
       │   Precision, Recall, F1       │
       │   Time, Efficiency            │
       └───────────────────────────────┘
```

---

## Slide 5: Analisi Sorgenti - Caratteristiche (1:30)

### Craigslist Dataset
- **Records**: [X]
- **Attributi**: [Y]
- **Valori nulli**: [Z]%
- **VIN coverage**: [W]%

### US Used Cars Dataset
- **Records**: [X]
- **Attributi**: [Y]
- **Valori nulli**: [Z]%
- **VIN coverage**: [W]%

**Key Insight**: Alta complementarità ma schema eterogeneo

---

## Slide 6: Quality Analysis (1:30)

### Top Attributi con Valori Nulli
[Grafico a barre orizzontale]

### Distribuzione Valori Unici
[Grafico a barre]

**Sfide identificate:**
- Missing values in attributi chiave
- Formato inconsistente (manufacturer, fuel, transmission)
- VIN parziali o rumorosi

---

## Slide 7: Schema Mediato (2:00)

### Attributi Integrati

| Categoria | Attributi |
|-----------|-----------|
| **Identificazione** | VIN, year, manufacturer, model |
| **Specifiche** | cylinders, fuel, transmission, drive |
| **Condizione** | condition, odometer, paint_color |
| **Prezzo** | price |
| **Localizzazione** | state |

### Normalizzazioni Applicate
- **Manufacturer**: `"chevy"` → `"chevrolet"`
- **Fuel**: `"gasoline"` → `"gas"`
- **Transmission**: `"auto"` → `"automatic"`
- **Price/Year**: Validazione range, outlier removal

---

## Slide 8: Ground Truth Generation (2:00)

### Strategia: VIN-based Matching

**Step 1: Pulizia VIN**
- Lunghezza minima: 11 caratteri
- Normalizzazione (uppercase, alphanumeric)
- Rimozione duplicati

**Step 2: Match Generation**
- VIN comuni → **Positive pairs**
- VIN diversi → **Negative pairs** (balanced)

**Step 3: Split**
- Training: **70%** ([X] coppie)
- Validation: **10%** ([Y] coppie)
- Test: **20%** ([Z] coppie)

**NOTA: VIN rimosso dopo generazione ground truth**

---

## Slide 9: Blocking Strategies (2:00)

### Perché Blocking?
**Prodotto cartesiano**: [N1] × [N2] = **[X] milioni** di confronti

### B1: Standard Blocking
**Keys**: `year` + `manufacturer`
- Reduction: **[X]%**
- Completeness: **[Y]%**
- Coppie: **[Z]**

### B2: Extended Blocking
**Keys**: `year` + `manufacturer` + `model_prefix(3)`
- Reduction: **[X]%**
- Completeness: **[Y]%**
- Coppie: **[Z]**

**Trade-off**: Maggiore riduzione vs rischio di missed matches

---

## Slide 10: Record Linkage - RecordLinkage (1:30)

### Approccio Rule-Based

**Similarity Functions:**
- `year`: Exact match (weight: 0.15)
- `manufacturer`: Jaro-Winkler ≥ 0.85 (weight: 0.20)
- `model`: Jaro-Winkler ≥ 0.85 (weight: 0.20)
- `price`: Numeric ± 10% (weight: 0.10)
- `odometer`: Numeric ± 15% (weight: 0.15)
- Altri: Exact/Categorical (weight: 0.20)

**Decision Rule:**
```
Match if weighted_score ≥ 0.7
```

**Pro**: Veloce, interpretabile
**Contro**: Threshold manuale, limitato

---

## Slide 11: Record Linkage - Dedupe (1:30)

### Approccio Machine Learning

**Libreria**: Python Dedupe
**Tipo**: Probabilistic record linkage + Active learning

**Features**:
- Exact: `year`
- String: `manufacturer`, `model`, `type`
- Price: `price`, `odometer`
- Categorical: `fuel`, `transmission`

**Training**:
- Supervised learning con ground truth
- [X] esempi match, [Y] esempi non-match
- Tempo training: **[Z] secondi**

**Pro**: Adattivo, apprendimento da dati
**Contro**: Training overhead

---

## Slide 12: Record Linkage - Ditto (1:30)

### Approccio Deep Learning

**Architettura**: Transformer-based (BERT-like)
**Repository**: FAIR-DA4ER

**Input Format**:
```
left: "year: 2015 [SEP] manufacturer: toyota [SEP] model: camry..."
right: "year: 2015 [SEP] manufacturer: toyota [SEP] model: camry..."
```

**Hyperparameters**:
- Max length: 256 tokens
- Batch size: 16
- Epochs: 20
- Learning rate: 3e-5

**Pro**: State-of-the-art accuracy
**Contro**: Computazionalmente costoso

---

## Slide 13: Valutazione - Setup (1:00)

### 6 Pipeline Testate

| Pipeline | Blocking | Matching Method |
|----------|----------|----------------|
| 1 | B1 | RecordLinkage |
| 2 | B2 | RecordLinkage |
| 3 | B1 | Dedupe |
| 4 | B2 | Dedupe |
| 5 | B1 | Ditto |
| 6 | B2 | Ditto |

### Metriche
- **Accuracy**: Precision, Recall, F1-Score
- **Efficiency**: Training Time, Inference Time

---

## Slide 14: Risultati - Performance (2:30)

### Metriche di Matching

[Grafico a barre con Precision, Recall, F1 per ogni pipeline]

**Top 3 F1-Score:**
1. **[Pipeline]**: [X] F1
2. **[Pipeline]**: [Y] F1
3. **[Pipeline]**: [Z] F1

**Key Findings:**
- [Metodo] supera gli altri in F1-score
- B1 vs B2: [Osservazione]
- Trade-off precision/recall: [Osservazione]

---

## Slide 15: Risultati - Efficiency (1:30)

### Tempi di Esecuzione

[Grafico a barre con Training Time e Inference Time]

**Fastest Training**: [Pipeline] - [X]s
**Fastest Inference**: [Pipeline] - [Y]s

**Key Findings:**
- Rule-based: Nessun training, inferenza veloce
- Dedupe: Training moderato, inferenza media
- Ditto: Training lungo, inferenza più lenta

**Trade-off**: Accuracy vs Computational cost

---

## Slide 16: Analisi Comparativa (1:30)

### Impatto del Blocking

**B1 (Standard)**
- PRO: Meno restrittivo
- PRO: Maggiore recall
- CONTRO: Più coppie da confrontare

**B2 (Extended)**
- PRO: Più selettivo
- PRO: Meno coppie (faster)
- CONTRO: Rischio missed matches

**Optimal**: Dipende da priorità (recall vs speed)

### Metodi di Match

**RecordLinkage**: Best per prototipazione rapida
**Dedupe**: Balance tra accuracy e costo
**Ditto**: Best accuracy, richiede risorse

---

## Slide 17: Confusion Matrix - Best Pipeline (1:00)

[Heatmap confusion matrix per la pipeline migliore]

### Analisi Errori

**False Positives**:
- [Analisi pattern comuni]

**False Negatives**:
- [Analisi pattern comuni]

**Possibili Miglioramenti**:
- Feature engineering avanzato
- Ensemble methods
- Threshold tuning

---

## Slide 18: Conclusioni (1:30)

### Risultati Principali

- **Schema mediato** integralmente definito e validato
- **Ground truth** di [X] coppie con [Y]% quality
- **6 pipeline** implementate e valutate
- **[Best method]** raggiunge **[X] F1-score**

### Sfide Superate

- Gestione rumore e missing values
- Scalabilità tramite blocking efficace
- Bilanciamento accuracy/efficiency

### Raccomandazioni

- **Production**: Usare [Metodo] con [Blocking]
- **Trade-off**: Considerare requisiti specifici
- **Futuro**: Ensemble methods, active learning

---

## Slide 19: Lavori Futuri (0:30)

### Possibili Estensioni

- **Feature Engineering**: Più attributi, embedding
- **Hyperparameter Optimization**: Grid/Bayesian search
- **Transfer Learning**: Modelli pre-trained su automotive domain
- **Active Learning**: Iterative improvement ground truth
- **Incremental Learning**: Adattamento a nuovi dati
- **Real-time Pipeline**: Streaming record linkage

---

## Slide 20: Q&A

# Domande?

**Contatti:**
- Email: [email]
- Repository: [GitHub link]
- Documentazione: [Link docs]

---

**Grazie per l'attenzione!**

---

# Note per il Presentatore

## Timing Suggerito
- Slide 1-3: 2:30 min
- Slide 4-7: 7:00 min
- Slide 8-12: 8:30 min
- Slide 13-17: 8:00 min
- Slide 18-20: 2:30 min
- **Buffer**: 1:30 min

## Key Messages
1. **Problema**: Integrazione dati eterogenei è sfida comune
2. **Soluzione**: Pipeline sistematica con blocking + matching
3. **Risultati**: Trade-off tra accuracy e efficiency ben documentato
4. **Valore**: Framework riusabile per altri domini

## Backup Slides
Preparare slide aggiuntive con:
- Dettagli tecnici implementazione
- Statistiche aggiuntive dataset
- Codice esempio
- Analisi errori dettagliata
