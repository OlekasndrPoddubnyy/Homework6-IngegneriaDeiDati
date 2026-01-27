# Esempio: Script Completo di Esecuzione Pipeline

"""
Questo script esegue l'intero workflow del progetto in sequenza.
Può essere usato come riferimento o per esecuzione automatizzata.
"""

import pandas as pd
import os
import yaml
import json
from pathlib import Path

# Import dei moduli custom
from src.data_analysis import DataAnalyzer, compare_sources
from src.schema_mediation import MediatedSchema, SourceAligner
from src.ground_truth import GroundTruthGenerator
from src.blocking import StandardBlocking, ExtendedBlocking, BlockingEvaluator
from src.record_linkage import RecordLinkageClassifier
from src.dedupe_model import DedupeModel
from src.ditto_model import DittoModel
from src.evaluation import PipelineEvaluator


def main():
    """Esegue il workflow completo"""
    
    print("="*80)
    print("AUTOMOBILE DATA INTEGRATION - RECORD LINKAGE PROJECT")
    print("="*80)
    
    # ==========================================
    # 1. CARICA CONFIGURAZIONE
    # ==========================================
    print("\n[1/8] Caricamento configurazione...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # ==========================================
    # 2. CARICA DATASET
    # ==========================================
    print("\n[2/8] Caricamento dataset...")
    
    # Craigslist
    craigslist_path = config['data']['sources']['craigslist']['path']
    df_craigslist = pd.read_csv(craigslist_path, low_memory=False)
    print(f"  Craigslist: {len(df_craigslist)} records, {len(df_craigslist.columns)} columns")
    
    # US Used Cars
    usedcars_path = config['data']['sources']['usedcars']['path']
    df_usedcars = pd.read_csv(usedcars_path, low_memory=False)
    print(f"  US Used Cars: {len(df_usedcars)} records, {len(df_usedcars.columns)} columns")
    
    # ==========================================
    # 3. ANALISI DATI
    # ==========================================
    print("\n[3/8] Analisi qualità dati...")
    
    analyzer1 = DataAnalyzer(df_craigslist, 'Craigslist')
    analyzer1.analyze_null_values()
    analyzer1.analyze_unique_values()
    analyzer1.get_summary_statistics()
    
    analyzer2 = DataAnalyzer(df_usedcars, 'US Used Cars')
    analyzer2.analyze_null_values()
    analyzer2.analyze_unique_values()
    analyzer2.get_summary_statistics()
    
    # Confronto
    comparison = compare_sources(analyzer1, analyzer2)
    print(comparison)
    
    # ==========================================
    # 4. SCHEMA MEDIATO E ALLINEAMENTO
    # ==========================================
    print("\n[4/8] Allineamento allo schema mediato...")
    
    schema = MediatedSchema('config.yaml')
    aligner = SourceAligner(schema)
    
    df1_aligned = aligner.align_craigslist(df_craigslist)
    df2_aligned = aligner.align_usedcars(df_usedcars)
    
    # Salva dati allineati
    os.makedirs('data/processed', exist_ok=True)
    df1_aligned.to_csv('data/processed/craigslist_aligned.csv', index=False)
    df2_aligned.to_csv('data/processed/usedcars_aligned.csv', index=False)
    
    print(f"  Dataset allineati salvati in data/processed/")
    
    # ==========================================
    # 5. GROUND TRUTH GENERATION
    # ==========================================
    print("\n[5/8] Generazione ground truth...")
    
    gt_generator = GroundTruthGenerator(
        df1_aligned, 
        df2_aligned,
        vin_column='vin',
        min_vin_length=config['ground_truth']['min_vin_length']
    )
    
    # Genera ground truth
    matches = gt_generator.find_matches()
    non_matches = gt_generator.generate_non_matches(ratio=1.0)
    ground_truth = gt_generator.create_ground_truth(include_features=True)
    
    # Salva (rimuovendo VIN)
    gt_generator.save_ground_truth(
        ground_truth,
        'data/ground_truth',
        remove_vin=True
    )
    
    print(f"  Ground truth salvata in data/ground_truth/")
    
    # ==========================================
    # 6. BLOCKING
    # ==========================================
    print("\n[6/8] Applicazione strategie di blocking...")
    
    # Carica dati senza VIN per blocking
    train = pd.read_csv('data/ground_truth/train.csv')
    test = pd.read_csv('data/ground_truth/test.csv')
    
    # Prepara dataframe per blocking (rimuovi VIN se presente)
    df1_for_blocking = df1_aligned.drop(columns=['vin'], errors='ignore')
    df2_for_blocking = df2_aligned.drop(columns=['vin'], errors='ignore')
    
    # B1: Standard Blocking
    b1 = StandardBlocking()
    pairs_b1 = b1.generate_pairs(df1_for_blocking, df2_for_blocking)
    
    # B2: Extended Blocking
    b2 = ExtendedBlocking()
    pairs_b2 = b2.generate_pairs(df1_for_blocking, df2_for_blocking)
    
    # Valuta blocking
    blocking_evaluator = BlockingEvaluator(test)
    metrics_b1 = blocking_evaluator.evaluate_blocking(pairs_b1, df1_for_blocking, df2_for_blocking)
    metrics_b2 = blocking_evaluator.evaluate_blocking(pairs_b2, df1_for_blocking, df2_for_blocking)
    
    print(f"  B1 - Coppie candidate: {len(pairs_b1):,}")
    print(f"  B2 - Coppie candidate: {len(pairs_b2):,}")
    
    # ==========================================
    # 7. RECORD LINKAGE
    # ==========================================
    print("\n[7/8] Esecuzione metodi di record linkage...")
    
    # Inizializza evaluator
    evaluator = PipelineEvaluator(test)
    
    # 7.1 RecordLinkage (Rule-based)
    print("\n  [7.1] RecordLinkage...")
    rl_classifier = RecordLinkageClassifier(config)
    
    results_b1_rl = rl_classifier.classify(df1_for_blocking, df2_for_blocking, pairs_b1)
    evaluator.evaluate_predictions(results_b1_rl, 'B1-RecordLinkage', df1_aligned, df2_aligned)
    
    results_b2_rl = rl_classifier.classify(df1_for_blocking, df2_for_blocking, pairs_b2)
    evaluator.evaluate_predictions(results_b2_rl, 'B2-RecordLinkage', df1_aligned, df2_aligned)
    
    # 7.2 Dedupe
    print("\n  [7.2] Dedupe...")
    dedupe_model = DedupeModel(num_cores=config['dedupe']['num_cores'])
    
    # Training
    training_time_b1 = dedupe_model.train(
        df1_for_blocking, 
        df2_for_blocking, 
        train,
        training_file='models/dedupe/training_b1.json'
    )
    
    # Predizione B1
    predictions_b1_dedupe, inference_time_b1 = dedupe_model.predict(
        df1_for_blocking, 
        df2_for_blocking, 
        pairs_b1
    )
    evaluator.evaluate_predictions(predictions_b1_dedupe, 'B1-Dedupe', df1_aligned, df2_aligned)
    evaluator.add_timing_info('B1-Dedupe', training_time_b1, inference_time_b1)
    
    # Predizione B2
    predictions_b2_dedupe, inference_time_b2 = dedupe_model.predict(
        df1_for_blocking, 
        df2_for_blocking, 
        pairs_b2
    )
    evaluator.evaluate_predictions(predictions_b2_dedupe, 'B2-Dedupe', df1_aligned, df2_aligned)
    evaluator.add_timing_info('B2-Dedupe', training_time_b1, inference_time_b2)
    
    # 7.3 Ditto
    print("\n  [7.3] Ditto...")
    print("  NOTA: Ditto richiede setup manuale del repository")
    print("  Vedere GETTING_STARTED.md per istruzioni")
    
    # Placeholder per Ditto (da implementare manualmente)
    # ditto_model = DittoModel()
    # ...
    
    # ==========================================
    # 8. VALUTAZIONE FINALE
    # ==========================================
    print("\n[8/8] Generazione report finale...")
    
    # Confronto pipeline
    comparison = evaluator.compare_pipelines()
    print("\n" + "="*80)
    print("CONFRONTO PIPELINE")
    print("="*80)
    print(comparison.to_string(index=False))
    
    # Genera visualizzazioni
    os.makedirs('results/visualizations', exist_ok=True)
    evaluator.plot_metrics_comparison('results/visualizations/metrics_comparison.png')
    evaluator.plot_timing_comparison('results/visualizations/timing_comparison.png')
    
    # Genera report
    report = evaluator.generate_report('results/evaluation_report.txt')
    
    print("\n" + "="*80)
    print("PROGETTO COMPLETATO!")
    print("="*80)
    print("\nFile generati:")
    print("  - data/processed/: Dataset allineati")
    print("  - data/ground_truth/: Ground truth train/val/test")
    print("  - models/: Modelli addestrati")
    print("  - results/: Report e visualizzazioni")
    print("\nProssimi passi:")
    print("  1. Implementare Ditto manualmente")
    print("  2. Completare docs/report.md con i risultati")
    print("  3. Preparare docs/presentation.md")


if __name__ == '__main__':
    main()
