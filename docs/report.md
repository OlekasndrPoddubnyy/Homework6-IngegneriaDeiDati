# Relazione Progetto: Integrazione Dati Automobili e Record Linkage

## 1. Introduzione

### 1.1 Obiettivi del Progetto
Il progetto si propone di integrare dati su automobili provenienti da diverse sorgenti online, affrontando le sfide tipiche dell'integrazione di dati eterogenei. Gli obiettivi principali sono:

1. Analizzare e caratterizzare le sorgenti dati
2. Definire uno schema mediato comune
3. Implementare strategie di record linkage
4. Valutare diverse pipeline di matching

### 1.2 Sorgenti Dati
Le sorgenti utilizzate sono:
- **Craigslist Cars & Trucks**: Dataset di annunci di veicoli usati
- **US Used Cars Dataset**: Dataset di veicoli usati dal mercato americano

### 1.3 Metodologia
Il progetto segue un approccio sistematico in 5 fasi:
1. Analisi delle sorgenti
2. Definizione schema mediato
3. Generazione ground truth
4. Implementazione strategie di blocking e matching
5. Valutazione sperimentale

---

## 2. Analisi delle Sorgenti Dati

### 2.1 Caratteristiche Generali

#### 2.1.1 Craigslist Dataset
- **Dimensioni**: [Da completare con analisi effettiva]
- **Attributi**: [Da completare]
- **Periodo temporale**: [Da completare]

#### 2.1.2 US Used Cars Dataset
- **Dimensioni**: [Da completare con analisi effettiva]
- **Attributi**: [Da completare]
- **Periodo temporale**: [Da completare]

### 2.2 Analisi Qualità dei Dati

#### 2.2.1 Valori Nulli
[Inserire tabelle e grafici con percentuali di valori nulli per attributo]

**Osservazioni principali:**
- Attributi con alta percentuale di valori nulli
- Differenze tra le sorgenti
- Impatto sulla strategia di integrazione

#### 2.2.2 Valori Unici
[Inserire tabelle e grafici con percentuali di valori unici]

**Osservazioni principali:**
- Cardinalità degli attributi
- Attributi candidati per blocking
- Distribuzione dei valori

### 2.3 Sfide Identificate

1. **Eterogeneità dello Schema**
   - Nomi attributi diversi
   - Unità di misura diverse
   - Formati diversi

2. **Qualità dei Dati**
   - Valori mancanti
   - Valori rumorosi
   - Duplicati

3. **VIN (Vehicle Identification Number)**
   - Presente in entrambe le sorgenti
   - Completezza variabile
   - Rumorosità dei dati

---

## 3. Schema Mediato

### 3.1 Definizione dello Schema

Lo schema mediato è stato definito considerando:
- Attributi comuni alle sorgenti
- Rilevanza per il matching
- Qualità dei dati

**Attributi dello schema mediato:**

| Attributo | Tipo | Descrizione | Sorgente 1 | Sorgente 2 |
|-----------|------|-------------|------------|------------|
| vin | String | Vehicle Identification Number | VIN | vin |
| year | Integer | Anno di produzione | year | year |
| manufacturer | String | Casa produttrice | manufacturer | make |
| model | String | Modello | model | model |
| price | Float | Prezzo | price | price |
| odometer | Float | Chilometraggio | odometer | mileage |
| fuel | String | Tipo carburante | fuel | fuel_type |
| transmission | String | Tipo trasmissione | transmission | transmission |
| type | String | Tipo veicolo | type | body_type |
| ... | ... | ... | ... | ... |

### 3.2 Normalizzazioni applicate

1. **Manufacturer**: Normalizzazione nomi (es. "chevy" → "chevrolet")
2. **Fuel Type**: Standardizzazione tipi carburante
3. **Transmission**: Categorizzazione (automatic/manual)
4. **Price**: Conversione in float, filtraggio outlier
5. **Year**: Validazione range (1900-2026)

---

## 4. Ground Truth Generation

### 4.1 Strategia

La ground truth è stata generata sfruttando l'attributo VIN:
- VIN univoco identifica lo stesso veicolo
- Pulizia dei VIN rumorosi
- Generazione automatica di coppie match/non-match

### 4.2 Pulizia dei Dati

**Criteri di filtraggio VIN:**
- Lunghezza minima: 11 caratteri
- Rimozione caratteri speciali
- Eliminazione duplicati

**Risultati:**
- Sorgente 1: [X] → [Y] VIN validi ([Z]% retention)
- Sorgente 2: [X] → [Y] VIN validi ([Z]% retention)
- VIN comuni: [N] coppie match

### 4.3 Bilanciamento Dataset

**Distribuzione Ground Truth:**
- Training: 70% ([X] coppie)
  - Match: [Y]
  - Non-match: [Z]
- Validation: 10% ([X] coppie)
- Test: 20% ([X] coppie)

---

## 5. Strategie di Blocking

### 5.1 B1: Standard Blocking

**Chiavi di blocking:**
- Year
- Manufacturer

**Razionale:**
- Attributi affidabili
- Bassa percentuale valori nulli
- Alta selettività

**Risultati:**
- Reduction Ratio: [X]%
- Pair Completeness: [Y]%
- Coppie candidate: [Z]

### 5.2 B2: Extended Blocking

**Chiavi di blocking:**
- Year
- Manufacturer
- Model (primi 3 caratteri)

**Razionale:**
- Maggiore selettività
- Riduzione ulteriore dello spazio di ricerca

**Risultati:**
- Reduction Ratio: [X]%
- Pair Completeness: [Y]%
- Coppie candidate: [Z]

### 5.3 Confronto Strategie

[Tabella comparativa con metriche]

---

## 6. Metodi di Record Linkage

### 6.1 Rule-Based (RecordLinkage Library)

**Funzioni di similarità utilizzate:**
- Year: Exact match
- Manufacturer: Jaro-Winkler (threshold: 0.85)
- Model: Jaro-Winkler (threshold: 0.85)
- Price: Numeric similarity (10% tolerance)
- Odometer: Numeric similarity (15% tolerance)

**Score pesato:**
```
Score = 0.15*year + 0.20*manufacturer + 0.20*model + 
        0.10*price + 0.15*odometer + 0.05*fuel + 
        0.05*transmission + 0.10*type
```

**Threshold decisionale:** 0.7

### 6.2 Dedupe

**Configurazione:**
- Campi: [Lista campi]
- Training: Supervised con ground truth
- Num cores: 4

**Training:**
- Esempi di training: [X] match, [Y] non-match
- Tempo di training: [Z] secondi

### 6.3 Ditto

**Configurazione:**
- Modello base: [Nome modello]
- Max length: 256 tokens
- Batch size: 16
- Epochs: 20
- Learning rate: 3e-5

**Training:**
- Architettura: Transformer-based
- Tempo di training: [X] secondi

---

## 7. Valutazione Sperimentale

### 7.1 Metriche di Valutazione

Per ogni pipeline sono state calcolate:
- **Precision**: Proporzione di match corretti tra quelli predetti
- **Recall**: Proporzione di match trovati rispetto ai veri match
- **F1-Score**: Media armonica di precision e recall
- **Training Time**: Tempo di addestramento del modello
- **Inference Time**: Tempo di predizione sui dati di test

### 7.2 Risultati

#### 7.2.1 Tabella Riepilogativa

| Pipeline | Precision | Recall | F1-Score | Training Time (s) | Inference Time (s) |
|----------|-----------|--------|----------|-------------------|-------------------|
| B1-RecordLinkage | [X] | [Y] | [Z] | N/A | [T] |
| B2-RecordLinkage | [X] | [Y] | [Z] | N/A | [T] |
| B1-Dedupe | [X] | [Y] | [Z] | [T1] | [T2] |
| B2-Dedupe | [X] | [Y] | [Z] | [T1] | [T2] |
| B1-Ditto | [X] | [Y] | [Z] | [T1] | [T2] |
| B2-Ditto | [X] | [Y] | [Z] | [T1] | [T2] |

#### 7.2.2 Grafici Comparativi

[Inserire grafici con metriche per pipeline]

### 7.3 Analisi dei Risultati

#### 7.3.1 Performance di Matching

**Migliore F1-Score:** [Nome pipeline] con [X]

**Osservazioni:**
- [Analisi delle performance]
- [Punti di forza e debolezza]
- [Trade-off precision/recall]

#### 7.3.2 Efficienza Computazionale

**Migliore efficienza:** [Nome pipeline]

**Osservazioni:**
- Confronto tempi rule-based vs ML
- Scalabilità delle soluzioni
- Trade-off accuracy/performance

#### 7.3.3 Impatto del Blocking

**B1 vs B2:**
- Riduzione coppie candidate: [Percentuale]
- Impatto su precision/recall: [Analisi]
- Trade-off reduction/completeness: [Analisi]

---

## 8. Conclusioni

### 8.1 Risultati Principali

1. **Schema Mediato:** Successfully integrato [X] attributi da [Y] sorgenti
2. **Ground Truth:** Generata con [X] coppie, [Y]% accuracy
3. **Blocking:** B2 riduce [X]% coppie mantenendo [Y]% completeness
4. **Matching:** [Metodo migliore] raggiunge [X] F1-score

### 8.2 Sfide Affrontate

1. **Rumore nei Dati:** Gestito con normalizzazioni e filtri
2. **Valori Mancanti:** Impatto mitigato con feature engineering
3. **Scalabilità:** Blocking riduce complessità computazionale
4. **Trade-off:** Bilanciamento precision/recall/efficiency

### 8.3 Raccomandazioni

**Per applicazioni production:**
- Utilizzare [Metodo] per [Scenario]
- Considerare ensemble di metodi
- Monitorare quality drift nel tempo

**Per futuri miglioramenti:**
- Feature engineering avanzato
- Hyperparameter tuning
- Active learning per ground truth
- Transfer learning con modelli pre-trained

---

## 9. Bibliografia

[Inserire riferimenti]

---

## Appendici

### Appendice A: Configurazioni Dettagliate
[Configurazioni YAML/JSON]

### Appendice B: Codice Sorgente
[Link al repository]

### Appendice C: Dataset e Risultati
[Link ai dati]
