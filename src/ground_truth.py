"""
Ground Truth Generation Module
Genera ground truth usando l'attributo VIN
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
import warnings


class GroundTruthGenerator:
    """Classe per generare ground truth usando VIN"""
    
    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                 vin_column: str = 'vin', 
                 min_vin_length: int = 11):
        """
        Args:
            df1: Primo DataFrame
            df2: Secondo DataFrame
            vin_column: Nome della colonna VIN
            min_vin_length: Lunghezza minima del VIN per considerarlo valido
        """
        self.df1 = df1.copy()
        self.df2 = df2.copy()
        self.vin_column = vin_column
        self.min_vin_length = min_vin_length
        
        # Aggiungi identificatori univoci
        self.df1['record_id'] = ['df1_' + str(i) for i in range(len(self.df1))]
        self.df2['record_id'] = ['df2_' + str(i) for i in range(len(self.df2))]
        
        self.matches = None
        self.non_matches = None
    
    def clean_vins(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pulisce i VIN rumorosi
        
        Returns:
            Tuple di DataFrame puliti
        """
        print("Pulizia VIN in corso...")
        
        # Filtra VIN validi in df1
        df1_valid = self.df1[
            self.df1[self.vin_column].notna() &
            (self.df1[self.vin_column].str.len() >= self.min_vin_length)
        ].copy()
        
        # Filtra VIN validi in df2
        df2_valid = self.df2[
            self.df2[self.vin_column].notna() &
            (self.df2[self.vin_column].str.len() >= self.min_vin_length)
        ].copy()
        
        # Rimuovi duplicati basati su VIN
        df1_valid = df1_valid.drop_duplicates(subset=[self.vin_column])
        df2_valid = df2_valid.drop_duplicates(subset=[self.vin_column])
        
        print(f"DF1: {len(self.df1)} -> {len(df1_valid)} record con VIN validi")
        print(f"DF2: {len(self.df2)} -> {len(df2_valid)} record con VIN validi")
        
        return df1_valid, df2_valid
    
    def find_matches(self) -> pd.DataFrame:
        """
        Trova coppie di record con VIN corrispondenti
        
        Returns:
            DataFrame con le coppie di match
        """
        df1_valid, df2_valid = self.clean_vins()
        
        # Trova VIN comuni
        common_vins = set(df1_valid[self.vin_column]) & set(df2_valid[self.vin_column])
        print(f"VIN comuni trovati: {len(common_vins)}")
        
        matches_list = []
        
        for vin in common_vins:
            # Trova record in df1
            records1 = df1_valid[df1_valid[self.vin_column] == vin]
            # Trova record in df2
            records2 = df2_valid[df2_valid[self.vin_column] == vin]
            
            # Crea tutte le coppie possibili (prodotto cartesiano)
            for _, rec1 in records1.iterrows():
                for _, rec2 in records2.iterrows():
                    matches_list.append({
                        'record_id_1': rec1['record_id'],
                        'record_id_2': rec2['record_id'],
                        'vin': vin,
                        'label': 1  # Match
                    })
        
        self.matches = pd.DataFrame(matches_list)
        print(f"Coppie di match generate: {len(self.matches)}")
        
        return self.matches
    
    def generate_non_matches(self, ratio: float = 1.0) -> pd.DataFrame:
        """
        Genera coppie di non-match
        
        Args:
            ratio: Rapporto non-matches/matches
            
        Returns:
            DataFrame con le coppie di non-match
        """
        if self.matches is None:
            self.find_matches()
        
        df1_valid, df2_valid = self.clean_vins()
        
        num_non_matches = int(len(self.matches) * ratio)
        print(f"Generazione di {num_non_matches} non-match...")
        
        non_matches_list = []
        attempts = 0
        max_attempts = num_non_matches * 10
        
        # Set di VIN match per evitarli
        match_vins = set(self.matches['vin'].unique())
        
        while len(non_matches_list) < num_non_matches and attempts < max_attempts:
            attempts += 1
            
            # Seleziona random record da df1
            rec1 = df1_valid.sample(1).iloc[0]
            # Seleziona random record da df2
            rec2 = df2_valid.sample(1).iloc[0]
            
            # Verifica che i VIN siano diversi
            if rec1[self.vin_column] != rec2[self.vin_column]:
                non_matches_list.append({
                    'record_id_1': rec1['record_id'],
                    'record_id_2': rec2['record_id'],
                    'vin_1': rec1[self.vin_column],
                    'vin_2': rec2[self.vin_column],
                    'label': 0  # Non-match
                })
        
        self.non_matches = pd.DataFrame(non_matches_list)
        print(f"Non-match generati: {len(self.non_matches)}")
        
        return self.non_matches
    
    def create_ground_truth(self, include_features: bool = True) -> pd.DataFrame:
        """
        Crea il ground truth completo combinando match e non-match
        
        Args:
            include_features: Se True, include anche i valori degli attributi
            
        Returns:
            DataFrame con ground truth completo
        """
        if self.matches is None:
            self.find_matches()
        
        if self.non_matches is None:
            self.generate_non_matches()
        
        # Combina match e non-match
        ground_truth = pd.concat([self.matches, self.non_matches], ignore_index=True)
        
        # Shuffle
        ground_truth = ground_truth.sample(frac=1, random_state=42).reset_index(drop=True)
        
        if include_features:
            ground_truth = self._add_features(ground_truth)
        
        print(f"Ground truth totale: {len(ground_truth)} coppie")
        print(f"  - Match: {(ground_truth['label'] == 1).sum()}")
        print(f"  - Non-match: {(ground_truth['label'] == 0).sum()}")
        
        return ground_truth
    
    def _add_features(self, ground_truth: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge i valori degli attributi al ground truth
        
        Args:
            ground_truth: DataFrame con le coppie
            
        Returns:
            DataFrame con feature aggiuntive
        """
        # Lista di attributi da includere (escluso VIN)
        feature_cols = [col for col in self.df1.columns 
                       if col not in [self.vin_column, 'record_id', 'source']]
        
        # Crea dizionari per lookup veloce
        df1_dict = self.df1.set_index('record_id').to_dict('index')
        df2_dict = self.df2.set_index('record_id').to_dict('index')
        
        # Aggiungi feature
        for col in feature_cols:
            ground_truth[f'{col}_1'] = ground_truth['record_id_1'].map(
                lambda x: df1_dict.get(x, {}).get(col)
            )
            ground_truth[f'{col}_2'] = ground_truth['record_id_2'].map(
                lambda x: df2_dict.get(x, {}).get(col)
            )
        
        return ground_truth
    
    def split_ground_truth(self, 
                          ground_truth: pd.DataFrame,
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide il ground truth in training, validation e test
        
        Args:
            ground_truth: DataFrame con ground truth
            test_size: Proporzione del test set
            val_size: Proporzione del validation set
            random_state: Seed per riproducibilità
            
        Returns:
            Tuple (train_df, val_df, test_df)
        """
        # Prima split: train+val vs test
        train_val, test = train_test_split(
            ground_truth,
            test_size=test_size,
            random_state=random_state,
            stratify=ground_truth['label']
        )
        
        # Seconda split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        
        # Verifica se ci sono abbastanza campioni per stratify
        min_samples_per_class = train_val['label'].value_counts().min()
        n_val = int(len(train_val) * val_size_adjusted)
        
        if n_val < 2 or min_samples_per_class < 2:
            # Se validation set troppo piccolo, usa split senza stratify
            train, val = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                random_state=random_state,
                shuffle=True
            )
        else:
            train, val = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=train_val['label']
            )
        
        print(f"Split completato:")
        print(f"  - Training: {len(train)} ({len(train)/len(ground_truth)*100:.1f}%)")
        print(f"  - Validation: {len(val)} ({len(val)/len(ground_truth)*100:.1f}%)")
        print(f"  - Test: {len(test)} ({len(test)/len(ground_truth)*100:.1f}%)")
        
        return train, val, test
    
    def verify_ground_truth_quality(self, ground_truth: pd.DataFrame) -> Dict:
        """
        Verifica la qualità del ground truth
        
        Args:
            ground_truth: DataFrame con ground truth
            
        Returns:
            Dizionario con statistiche di qualità
        """
        stats = {
            'total_pairs': len(ground_truth),
            'match_pairs': (ground_truth['label'] == 1).sum(),
            'non_match_pairs': (ground_truth['label'] == 0).sum(),
            'match_ratio': (ground_truth['label'] == 1).mean(),
            'duplicates': ground_truth.duplicated(subset=['record_id_1', 'record_id_2']).sum(),
        }
        
        # Verifica distribuzione degli attributi
        feature_cols = [col for col in ground_truth.columns if col.endswith('_1')]
        
        for col in feature_cols:
            base_col = col[:-2]  # Rimuovi '_1'
            col_1 = f'{base_col}_1'
            col_2 = f'{base_col}_2'
            
            if col_1 in ground_truth.columns and col_2 in ground_truth.columns:
                # Calcola percentuale di valori nulli
                null_perc_1 = ground_truth[col_1].isna().mean() * 100
                null_perc_2 = ground_truth[col_2].isna().mean() * 100
                
                stats[f'{base_col}_null_perc'] = (null_perc_1 + null_perc_2) / 2
        
        return stats
    
    def save_ground_truth(self, 
                         ground_truth: pd.DataFrame,
                         output_dir: str,
                         remove_vin: bool = True):
        """
        Salva il ground truth su file
        
        Args:
            ground_truth: DataFrame con ground truth
            output_dir: Directory di output
            remove_vin: Se True, rimuove le colonne VIN
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Rimuovi VIN se richiesto
        if remove_vin:
            vin_cols = [col for col in ground_truth.columns if 'vin' in col.lower()]
            ground_truth = ground_truth.drop(columns=vin_cols, errors='ignore')
            print(f"Colonne VIN rimosse: {vin_cols}")
        
        # Split e salva
        train, val, test = self.split_ground_truth(ground_truth)
        
        train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val.to_csv(os.path.join(output_dir, 'validation.csv'), index=False)
        test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        # Salva anche il ground truth completo
        ground_truth.to_csv(os.path.join(output_dir, 'ground_truth_full.csv'), index=False)
        
        print(f"Ground truth salvato in {output_dir}")
