"""
Dedupe Model Module
Implementa modello di record linkage usando Dedupe
"""

import pandas as pd
import dedupe
import os
import json
from typing import Dict, List, Tuple
import time


class DedupeModel:
    """Wrapper per la libreria Dedupe"""
    
    def __init__(self, fields: List[Dict] = None, num_cores: int = 4):
        """
        Args:
            fields: Lista di definizioni dei campi per Dedupe
            num_cores: Numero di core da usare
        """
        self.fields = fields or self._default_fields()
        self.num_cores = num_cores
        self.linker = None
        self.training_pairs = None
    
    def _default_fields(self) -> List[Dict]:
        """
        Definizione di default dei campi per Dedupe
        
        Returns:
            Lista di definizioni dei campi
        """
        return [
            {'field': 'year', 'type': 'Exact'},
            {'field': 'manufacturer', 'type': 'String'},
            {'field': 'model', 'type': 'String'},
            {'field': 'price', 'type': 'Price'},
            {'field': 'odometer', 'type': 'Price'},  # Usa Price per numeric
            {'field': 'fuel', 'type': 'Categorical', 'categories': ['gas', 'diesel', 'electric', 'hybrid']},
            {'field': 'transmission', 'type': 'Categorical', 'categories': ['automatic', 'manual']},
            {'field': 'type', 'type': 'String'}
        ]
    
    def prepare_data(self, df: pd.DataFrame) -> Dict:
        """
        Prepara i dati nel formato richiesto da Dedupe
        
        Args:
            df: DataFrame da preparare
            
        Returns:
            Dizionario con i dati preparati
        """
        data_dict = {}
        
        for idx, row in df.iterrows():
            record = {}
            for field_def in self.fields:
                field_name = field_def['field']
                if field_name in df.columns and pd.notna(row[field_name]):
                    record[field_name] = str(row[field_name])
            
            data_dict[idx] = record
        
        return data_dict
    
    def prepare_training_data(self, ground_truth: pd.DataFrame,
                             df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """
        Prepara dati di training dal ground truth
        
        Args:
            ground_truth: DataFrame con ground truth
            df1: Primo DataFrame
            df2: Secondo DataFrame
            
        Returns:
            Dizionario con training examples
        """
        training_examples = {
            'match': [],
            'distinct': []
        }
        
        # Converti DataFrames in dizionari
        data1 = self.prepare_data(df1)
        data2 = self.prepare_data(df2)
        
        # Crea mapping record_id -> index
        id_to_idx1 = {row['record_id']: idx for idx, row in df1.iterrows() if 'record_id' in df1.columns}
        id_to_idx2 = {row['record_id']: idx for idx, row in df2.iterrows() if 'record_id' in df2.columns}
        
        for _, row in ground_truth.iterrows():
            idx1 = id_to_idx1.get(row.get('record_id_1'))
            idx2 = id_to_idx2.get(row.get('record_id_2'))
            
            if idx1 is not None and idx2 is not None:
                record1 = data1.get(idx1)
                record2 = data2.get(idx2)
                
                if record1 and record2:
                    if row['label'] == 1:
                        training_examples['match'].append((record1, record2))
                    else:
                        training_examples['distinct'].append((record1, record2))
        
        self.training_pairs = training_examples
        
        print(f"Training examples preparati:")
        print(f"  - Match: {len(training_examples['match'])}")
        print(f"  - Distinct: {len(training_examples['distinct'])}")
        
        return training_examples
    
    def train(self, data1: pd.DataFrame, data2: pd.DataFrame,
             ground_truth: pd.DataFrame = None,
             training_file: str = None) -> float:
        """
        Addestra il modello Dedupe
        
        Args:
            data1: Primo DataFrame
            data2: Secondo DataFrame
            ground_truth: DataFrame con ground truth (opzionale)
            training_file: File con esempi di training salvati
            
        Returns:
            Tempo di training in secondi
        """
        start_time = time.time()
        
        # Prepara dati
        data_dict1 = self.prepare_data(data1)
        data_dict2 = self.prepare_data(data2)
        
        print("Inizializzazione del modello Dedupe...")
        
        # Crea RecordLink object
        self.linker = dedupe.RecordLink(self.fields, num_cores=self.num_cores)
        
        # Carica o prepara training data
        if training_file and os.path.exists(training_file):
            print(f"Caricamento training data da {training_file}")
            with open(training_file, 'r') as f:
                self.linker.prepare_training(data_dict1, data_dict2, 
                                            training_file=training_file)
        else:
            print("Preparazione training data...")
            self.linker.prepare_training(data_dict1, data_dict2)
            
            # Se abbiamo ground truth, usalo
            if ground_truth is not None:
                print("Uso del ground truth per il training...")
                training_examples = self.prepare_training_data(ground_truth, data1, data2)
                
                # Mark labeled pairs
                for record1, record2 in training_examples['match']:
                    self.linker.mark_pairs({'match': [(record1, record2)]})
                
                for record1, record2 in training_examples['distinct']:
                    self.linker.mark_pairs({'distinct': [(record1, record2)]})
        
        # Train
        print("Training del modello in corso...")
        self.linker.train()
        
        # Save training data
        if training_file:
            os.makedirs(os.path.dirname(training_file), exist_ok=True)
            with open(training_file, 'w') as f:
                self.linker.write_training(f)
            print(f"Training data salvati in {training_file}")
        
        training_time = time.time() - start_time
        print(f"Training completato in {training_time:.2f} secondi")
        
        return training_time
    
    def predict(self, data1: pd.DataFrame, data2: pd.DataFrame,
               candidate_pairs: pd.MultiIndex = None,
               threshold: float = 0.5) -> Tuple[pd.DataFrame, float]:
        """
        Predice match usando il modello addestrato
        
        Args:
            data1: Primo DataFrame
            data2: Secondo DataFrame
            candidate_pairs: Coppie candidate (se None, usa tutte le coppie)
            threshold: Soglia di confidence per i match
            
        Returns:
            Tuple (DataFrame con i match, tempo di inferenza)
        """
        if self.linker is None:
            raise ValueError("Model not trained. Run train() first.")
        
        start_time = time.time()
        
        # Prepara dati
        data_dict1 = self.prepare_data(data1)
        data_dict2 = self.prepare_data(data2)
        
        print("Predizione dei match in corso...")
        
        # Link records
        linked_records = self.linker.join(data_dict1, data_dict2, threshold)
        
        # Converti in DataFrame
        matches = []
        for cluster_id, (records, scores) in enumerate(linked_records):
            for (idx1, idx2), score in zip(records, scores):
                matches.append({
                    'index_1': idx1,
                    'index_2': idx2,
                    'cluster_id': cluster_id,
                    'confidence': score,
                    'prediction': 1
                })
        
        matches_df = pd.DataFrame(matches)
        
        inference_time = time.time() - start_time
        
        print(f"Predizione completata in {inference_time:.2f} secondi")
        print(f"Match trovati: {len(matches_df)}")
        
        return matches_df, inference_time
    
    def save_model(self, settings_file: str):
        """
        Salva il modello addestrato
        
        Args:
            settings_file: Percorso dove salvare il modello
        """
        if self.linker is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        
        with open(settings_file, 'wb') as f:
            self.linker.write_settings(f)
        
        print(f"Modello salvato in {settings_file}")
    
    def load_model(self, settings_file: str):
        """
        Carica un modello salvato
        
        Args:
            settings_file: Percorso del modello salvato
        """
        if not os.path.exists(settings_file):
            raise FileNotFoundError(f"Settings file not found: {settings_file}")
        
        print(f"Caricamento modello da {settings_file}")
        
        with open(settings_file, 'rb') as f:
            self.linker = dedupe.StaticRecordLink(f, num_cores=self.num_cores)
        
        print("Modello caricato con successo")


class DedupeEvaluator:
    """Classe per valutare le performance di Dedupe"""
    
    @staticmethod
    def evaluate_predictions(predictions: pd.DataFrame, 
                           ground_truth: pd.DataFrame,
                           df1: pd.DataFrame,
                           df2: pd.DataFrame) -> Dict:
        """
        Valuta le predizioni rispetto al ground truth
        
        Args:
            predictions: DataFrame con le predizioni
            ground_truth: DataFrame con ground truth
            df1: Primo DataFrame
            df2: Secondo DataFrame
            
        Returns:
            Dizionario con metriche di valutazione
        """
        # Crea mapping indici -> record_id
        idx_to_id1 = {idx: row['record_id'] for idx, row in df1.iterrows() if 'record_id' in df1.columns}
        idx_to_id2 = {idx: row['record_id'] for idx, row in df2.iterrows() if 'record_id' in df2.columns}
        
        # Crea set di coppie predette
        predicted_pairs = set()
        for _, row in predictions.iterrows():
            id1 = idx_to_id1.get(row['index_1'])
            id2 = idx_to_id2.get(row['index_2'])
            if id1 and id2:
                predicted_pairs.add((id1, id2))
        
        # Crea set di coppie vere
        true_pairs = set()
        for _, row in ground_truth[ground_truth['label'] == 1].iterrows():
            true_pairs.add((row['record_id_1'], row['record_id_2']))
        
        # Calcola metriche
        true_positives = len(predicted_pairs & true_pairs)
        false_positives = len(predicted_pairs - true_pairs)
        false_negatives = len(true_pairs - predicted_pairs)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_predictions': len(predicted_pairs),
            'total_true_pairs': len(true_pairs)
        }
        
        return metrics
