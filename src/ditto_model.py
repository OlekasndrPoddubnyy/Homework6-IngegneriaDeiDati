"""
Ditto Model Module
Implementa modello di record linkage usando Ditto
"""

import pandas as pd
import numpy as np
import os
import json
import subprocess
from typing import Dict, List, Tuple
import time


class DittoModel:
    """Wrapper per il modello Ditto (Deep Learning based Entity Matching)"""
    
    def __init__(self, model_name: str = 'ditto', 
                 max_len: int = 256,
                 batch_size: int = 16,
                 epochs: int = 20,
                 learning_rate: float = 3e-5,
                 checkpoint_dir: str = 'models/ditto'):
        """
        Args:
            model_name: Nome del modello base
            max_len: Lunghezza massima delle sequenze
            batch_size: Batch size per training
            epochs: Numero di epoch
            learning_rate: Learning rate
            checkpoint_dir: Directory per i checkpoint
        """
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.ditto_repo_path = None
    
    def setup_ditto(self, repo_url: str = 'https://github.com/MarcoNapoleone/FAIR-DA4ER'):
        """
        Clona e configura il repository Ditto
        
        Args:
            repo_url: URL del repository Ditto
        """
        print(f"Setup Ditto repository da {repo_url}")
        
        # Clone repository se non esiste
        repo_name = repo_url.split('/')[-1]
        if not os.path.exists(repo_name):
            print(f"Clonazione repository...")
            subprocess.run(['git', 'clone', repo_url], check=True)
        
        self.ditto_repo_path = repo_name
        print(f"Ditto configurato in {self.ditto_repo_path}")
    
    def prepare_data_for_ditto(self, ground_truth: pd.DataFrame,
                               df1: pd.DataFrame, df2: pd.DataFrame,
                               output_dir: str) -> Dict[str, str]:
        """
        Prepara i dati nel formato richiesto da Ditto
        
        Args:
            ground_truth: DataFrame con ground truth
            df1: Primo DataFrame
            df2: Secondo DataFrame
            output_dir: Directory di output
            
        Returns:
            Dizionario con i percorsi dei file creati
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Crea mapping record_id -> dati
        id_to_data1 = {}
        for idx, row in df1.iterrows():
            if 'record_id' in df1.columns:
                id_to_data1[row['record_id']] = row.to_dict()
        
        id_to_data2 = {}
        for idx, row in df2.iterrows():
            if 'record_id' in df2.columns:
                id_to_data2[row['record_id']] = row.to_dict()
        
        # Prepara dati in formato Ditto (formato CSV o JSON)
        ditto_data = []
        
        for _, row in ground_truth.iterrows():
            rec1_id = row.get('record_id_1')
            rec2_id = row.get('record_id_2')
            label = row['label']
            
            rec1 = id_to_data1.get(rec1_id, {})
            rec2 = id_to_data2.get(rec2_id, {})
            
            # Crea rappresentazione testuale dei record
            text1 = self._record_to_text(rec1)
            text2 = self._record_to_text(rec2)
            
            ditto_data.append({
                'left': text1,
                'right': text2,
                'label': label
            })
        
        # Salva in formato CSV
        ditto_df = pd.DataFrame(ditto_data)
        output_file = os.path.join(output_dir, 'ditto_data.csv')
        ditto_df.to_csv(output_file, index=False)
        
        print(f"Dati Ditto salvati in {output_file}")
        print(f"  - Totale coppie: {len(ditto_df)}")
        print(f"  - Match: {(ditto_df['label'] == 1).sum()}")
        print(f"  - Non-match: {(ditto_df['label'] == 0).sum()}")
        
        return {'data_file': output_file}
    
    def _record_to_text(self, record: Dict) -> str:
        """
        Converte un record in rappresentazione testuale
        
        Args:
            record: Dizionario con i dati del record
            
        Returns:
            Stringa con rappresentazione testuale
        """
        # Seleziona campi importanti
        important_fields = ['year', 'manufacturer', 'model', 'price', 'odometer', 
                          'fuel', 'transmission', 'type', 'condition']
        
        parts = []
        for field in important_fields:
            value = record.get(field)
            if pd.notna(value) and value != '':
                parts.append(f"{field}: {value}")
        
        return " [SEP] ".join(parts)
    
    def train(self, train_file: str, val_file: str = None) -> float:
        """
        Addestra il modello Ditto
        
        Args:
            train_file: File con dati di training
            val_file: File con dati di validation (opzionale)
            
        Returns:
            Tempo di training in secondi
        """
        if self.ditto_repo_path is None:
            raise ValueError("Ditto repository not setup. Run setup_ditto() first.")
        
        start_time = time.time()
        
        print(f"Training Ditto model...")
        print(f"  - Training file: {train_file}")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Learning rate: {self.learning_rate}")
        
        # Comando per training (da adattare al repository specifico)
        # Questo è un esempio generico, va adattato al repository effettivo
        train_command = [
            'python', 
            os.path.join(self.ditto_repo_path, 'train.py'),
            '--train_file', train_file,
            '--model_name', self.model_name,
            '--epochs', str(self.epochs),
            '--batch_size', str(self.batch_size),
            '--lr', str(self.learning_rate),
            '--max_len', str(self.max_len),
            '--output_dir', self.checkpoint_dir
        ]
        
        if val_file:
            train_command.extend(['--val_file', val_file])
        
        # NOTA: Questo è un placeholder. Il comando effettivo dipende dall'implementazione
        # del repository Ditto specifico
        print(f"Comando training: {' '.join(train_command)}")
        print("NOTA: Adattare il comando al repository Ditto specifico")
        
        # subprocess.run(train_command, check=True)
        
        training_time = time.time() - start_time
        print(f"Training completato in {training_time:.2f} secondi")
        
        return training_time
    
    def predict(self, test_file: str, 
               candidate_pairs: pd.MultiIndex = None) -> Tuple[pd.DataFrame, float]:
        """
        Predice match usando il modello addestrato
        
        Args:
            test_file: File con dati di test
            candidate_pairs: Coppie candidate (opzionale)
            
        Returns:
            Tuple (DataFrame con predizioni, tempo di inferenza)
        """
        if not os.path.exists(self.checkpoint_dir):
            raise ValueError(f"Model checkpoint not found: {self.checkpoint_dir}")
        
        start_time = time.time()
        
        print(f"Predizione con Ditto model...")
        
        # Comando per predizione (da adattare)
        predict_command = [
            'python',
            os.path.join(self.ditto_repo_path, 'predict.py'),
            '--test_file', test_file,
            '--model_dir', self.checkpoint_dir,
            '--batch_size', str(self.batch_size),
            '--max_len', str(self.max_len)
        ]
        
        print(f"Comando predizione: {' '.join(predict_command)}")
        print("NOTA: Adattare il comando al repository Ditto specifico")
        
        # subprocess.run(predict_command, check=True)
        
        # Carica predizioni (il formato dipende dall'output di Ditto)
        # Questo è un placeholder
        predictions_file = os.path.join(self.checkpoint_dir, 'predictions.csv')
        
        # predictions_df = pd.read_csv(predictions_file)
        # Per ora, crea un DataFrame vuoto come placeholder
        predictions_df = pd.DataFrame(columns=['index_1', 'index_2', 'confidence', 'prediction'])
        
        inference_time = time.time() - start_time
        
        print(f"Predizione completata in {inference_time:.2f} secondi")
        print(f"Predizioni: {len(predictions_df)}")
        
        return predictions_df, inference_time
    
    def prepare_inference_data(self, df1: pd.DataFrame, df2: pd.DataFrame,
                              candidate_pairs: pd.MultiIndex,
                              output_file: str):
        """
        Prepara dati per inferenza
        
        Args:
            df1: Primo DataFrame
            df2: Secondo DataFrame
            candidate_pairs: Coppie candidate
            output_file: File di output
        """
        inference_data = []
        
        for idx1, idx2 in candidate_pairs:
            rec1 = df1.loc[idx1].to_dict()
            rec2 = df2.loc[idx2].to_dict()
            
            text1 = self._record_to_text(rec1)
            text2 = self._record_to_text(rec2)
            
            inference_data.append({
                'index_1': idx1,
                'index_2': idx2,
                'left': text1,
                'right': text2
            })
        
        inference_df = pd.DataFrame(inference_data)
        inference_df.to_csv(output_file, index=False)
        
        print(f"Dati per inferenza salvati in {output_file}")
        print(f"  - Coppie da predire: {len(inference_df)}")


class DittoIntegrationGuide:
    """Guida per l'integrazione con Ditto"""
    
    @staticmethod
    def print_setup_instructions():
        """Stampa istruzioni per setup di Ditto"""
        instructions = """
        ========================================
        ISTRUZIONI PER INTEGRARE DITTO
        ========================================
        
        1. Clonare il repository:
           git clone https://github.com/MarcoNapoleone/FAIR-DA4ER
        
        2. Installare dipendenze:
           cd FAIR-DA4ER
           pip install -r requirements.txt
        
        3. Preparare i dati nel formato richiesto:
           - Il modulo DittoModel fornisce metodi per preparare i dati
           - prepare_data_for_ditto() crea i file CSV necessari
        
        4. Training:
           - Usare il metodo train() di DittoModel
           - Oppure eseguire manualmente lo script di training del repository
        
        5. Predizione:
           - Usare il metodo predict() di DittoModel
           - Oppure eseguire manualmente lo script di predizione
        
        6. Note importanti:
           - Verificare la struttura esatta del repository clonato
           - Adattare i comandi ai nomi degli script effettivi
           - Verificare il formato di input/output richiesto
        
        ========================================
        """
        print(instructions)
    
    @staticmethod
    def create_training_script(output_path: str = 'train_ditto.py'):
        """
        Crea uno script di esempio per training con Ditto
        
        Args:
            output_path: Percorso dello script da creare
        """
        script = """
# Script di esempio per training con Ditto
# Adattare in base al repository specifico

import os
import sys

# Aggiungere il path del repository Ditto
sys.path.append('FAIR-DA4ER')

# Importare i moduli necessari dal repository Ditto
# from ditto import train_model

def train_ditto_model(train_file, val_file, output_dir):
    \"\"\"
    Addestra un modello Ditto
    \"\"\"
    print(f"Training Ditto model...")
    print(f"Train file: {train_file}")
    print(f"Val file: {val_file}")
    print(f"Output: {output_dir}")
    
    # Configurazione
    config = {
        'batch_size': 16,
        'epochs': 20,
        'learning_rate': 3e-5,
        'max_len': 256
    }
    
    # Training (da implementare con le API di Ditto)
    # model = train_model(train_file, val_file, config)
    # model.save(output_dir)
    
    print("Training completato!")

if __name__ == '__main__':
    train_ditto_model(
        train_file='data/ground_truth/train.csv',
        val_file='data/ground_truth/validation.csv',
        output_dir='models/ditto'
    )
"""
        
        with open(output_path, 'w') as f:
            f.write(script)
        
        print(f"Script di training creato: {output_path}")
