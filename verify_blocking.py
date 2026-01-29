"""
Script per verificare se le coppie di ground truth sono presenti nei candidate pairs del blocking
e se vengono filtrate dalla soglia di similarità.
"""

import pandas as pd
import os

print("=" * 80)
print("VERIFICA BLOCKING STRATEGY")
print("=" * 80)

# Carica ground truth
gt_path = "data/ground_truth/ground_truth_full.csv"
print(f"\n1. Caricamento ground truth da: {gt_path}")
gt_df = pd.read_csv(gt_path)

# Estrai solo le coppie match (label=1)
match_pairs = gt_df[gt_df['label'] == 1]
print(f"   Trovate {len(match_pairs)} coppie di match nel ground truth")

# Mostra i dettagli delle coppie match
print("\n   Dettagli delle coppie match:")
for idx, row in match_pairs.iterrows():
    print(f"   - {row['record_id_1']} <-> {row['record_id_2']}")
    print(f"     VIN: {row['vin']}")
    print(f"     Year: {row['year_1']} - {row['year_2']}")
    print(f"     Manufacturer: {row['manufacturer_1']} - {row['manufacturer_2']}")
    print(f"     Model: {row['model_1']} - {row['model_2']}")
    print()

# Carica i dataset campionati per mappare record_id a indici
print("2. Caricamento dataset campionati per mappatura indici...")
craigslist_df = pd.read_csv("data/processed/craigslist_sample.csv")
usedcars_df = pd.read_csv("data/processed/usedcars_sample.csv")

print(f"   Craigslist: {len(craigslist_df)} record")
print(f"   UsedCars: {len(usedcars_df)} record")

# Crea mappatura record_id -> indice (il record_id già esiste nel CSV)
print("\n3. Creazione mappatura record_id -> indice...")

craigslist_id_to_idx = {rid: idx for idx, rid in enumerate(craigslist_df['record_id'])}
usedcars_id_to_idx = {rid: idx for idx, rid in enumerate(usedcars_df['record_id'])}

print(f"   Mappati {len(craigslist_id_to_idx)} record Craigslist")
print(f"   Mappati {len(usedcars_id_to_idx)} record UsedCars")

# Converte le coppie ground truth in indici
print("\n4. Conversione coppie ground truth in indici...")
gt_pairs_as_indices = []
for idx, row in match_pairs.iterrows():
    rid1 = row['record_id_1']
    rid2 = row['record_id_2']
    
    if rid1 in craigslist_id_to_idx and rid2 in usedcars_id_to_idx:
        idx1 = craigslist_id_to_idx[rid1]
        idx2 = usedcars_id_to_idx[rid2]
        gt_pairs_as_indices.append((idx1, idx2))
        print(f"   {rid1} -> idx {idx1}")
        print(f"   {rid2} -> idx {idx2}")
    else:
        print(f"   ERRORE: {rid1} o {rid2} non trovato nei dataset campionati!")

print(f"\n   Totale coppie convertite: {len(gt_pairs_as_indices)}")

# Verifica se le coppie sono nei candidate pairs B1
print("\n5. Verifica presenza nei candidate pairs B1...")
b1_path = "results/b1_candidate_pairs.csv"

if os.path.exists(b1_path):
    # Leggi il file in chunks per gestire dimensioni grandi
    print(f"   Lettura file B1: {b1_path}")
    chunk_size = 100000
    found_pairs = []
    total_pairs = 0
    
    for chunk in pd.read_csv(b1_path, chunksize=chunk_size):
        total_pairs += len(chunk)
        
        for idx1, idx2 in gt_pairs_as_indices:
            # Cerca la coppia nel chunk
            match = chunk[(chunk['craigslist_idx'] == idx1) & (chunk['usedcars_idx'] == idx2)]
            if not match.empty:
                found_pairs.append({
                    'craigslist_idx': idx1,
                    'usedcars_idx': idx2,
                    'found': True
                })
    
    print(f"   Totale candidate pairs B1: {total_pairs:,}")
    print(f"   Coppie ground truth trovate in B1: {len(found_pairs)}/{len(gt_pairs_as_indices)}")
    
    if len(found_pairs) < len(gt_pairs_as_indices):
        print("\n   ⚠️ PROBLEMA: Non tutte le coppie ground truth sono nel blocking!")
        print("   Questo significa che il blocking sta filtrando troppo aggressivamente.")
        
        # Identifica quali coppie mancano
        found_indices = set((p['craigslist_idx'], p['usedcars_idx']) for p in found_pairs)
        missing_pairs = [p for p in gt_pairs_as_indices if p not in found_indices]
        
        print(f"\n   Coppie mancanti dal blocking:")
        for idx1, idx2 in missing_pairs:
            # Trova i record_id corrispondenti
            rid1 = craigslist_df.iloc[idx1]['record_id']
            rid2 = usedcars_df.iloc[idx2]['record_id']
            
            # Recupera i dati per capire perché non sono nel blocking
            r1 = craigslist_df.iloc[idx1]
            r2 = usedcars_df.iloc[idx2]
            
            print(f"   - {rid1} <-> {rid2}")
            print(f"     Craigslist: year={r1['year']}, manufacturer={r1['manufacturer']}")
            print(f"     UsedCars: year={r2['year']}, manufacturer={r2['manufacturer']}")
            print(f"     → Chiavi diverse: il blocking B1 richiede year+manufacturer identici")
    else:
        print("\n   ✅ Tutte le coppie ground truth sono presenti nel blocking!")
        print("   Il problema deve essere nella fase di classificazione (threshold troppo alta).")
else:
    print(f"   ERRORE: File {b1_path} non trovato!")

# Verifica i punteggi nelle predizioni di Phase 5
print("\n6. Verifica punteggi nelle predizioni Phase 5...")
predictions_path = "results/b1_record_linkage_predictions.csv"

if os.path.exists(predictions_path):
    print(f"   Lettura predizioni: {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)
    
    print(f"   Totale predizioni: {len(predictions_df)}")
    print(f"   Predizioni con match=True: {predictions_df['match'].sum()}")
    
    # Cerca le coppie ground truth nelle predizioni
    found_in_predictions = 0
    for idx1, idx2 in gt_pairs_as_indices:
        match = predictions_df[
            (predictions_df['craigslist_idx'] == idx1) & 
            (predictions_df['usedcars_idx'] == idx2)
        ]
        
        if not match.empty:
            found_in_predictions += 1
            score = match.iloc[0]['similarity_score']
            is_match = match.iloc[0]['match']
            
            # Trova i record_id
            rid1 = craigslist_df.iloc[idx1]['record_id']
            rid2 = usedcars_df.iloc[idx2]['record_id']
            
            print(f"\n   Coppia: {rid1} <-> {rid2}")
            print(f"   - Similarity Score: {score:.4f}")
            print(f"   - Classificato come Match: {is_match}")
            print(f"   - Threshold: 0.7")
            
            if not is_match:
                print(f"   ⚠️ Score sotto soglia! ({score:.4f} < 0.7)")
    
    print(f"\n   Coppie ground truth trovate nelle predizioni: {found_in_predictions}/{len(gt_pairs_as_indices)}")
    
    if found_in_predictions == 0:
        print("\n   ⚠️ NESSUNA coppia ground truth trovata nelle predizioni!")
        print("   Questo indica che le coppie non sono nemmeno arrivate alla fase di scoring.")
else:
    print(f"   ERRORE: File {predictions_path} non trovato!")

print("\n" + "=" * 80)
print("FINE VERIFICA")
print("=" * 80)
