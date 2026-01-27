"""
Blocking Strategies Module
Implementa strategie di blocking per ridurre lo spazio di confronto
"""

import pandas as pd
from typing import List, Tuple, Set
import recordlinkage as rl
from itertools import product


class BlockingStrategy:
    """Classe base per strategie di blocking"""
    
    def __init__(self, name: str):
        """
        Args:
            name: Nome della strategia
        """
        self.name = name
        self.candidate_pairs = None
    
    def generate_pairs(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.MultiIndex:
        """
        Genera coppie candidate
        
        Args:
            df1: Primo DataFrame
            df2: Secondo DataFrame
            
        Returns:
            MultiIndex con le coppie candidate
        """
        raise NotImplementedError
    
    def get_statistics(self) -> dict:
        """
        Ritorna statistiche sulla strategia di blocking
        
        Returns:
            Dizionario con statistiche
        """
        if self.candidate_pairs is None:
            return {'error': 'No candidate pairs generated'}
        
        return {
            'strategy_name': self.name,
            'total_candidate_pairs': len(self.candidate_pairs),
            'reduction_ratio': self._calculate_reduction_ratio()
        }
    
    def _calculate_reduction_ratio(self) -> float:
        """
        Calcola il ratio di riduzione rispetto al prodotto cartesiano
        
        Returns:
            Ratio di riduzione
        """
        # Questa funzione va implementata nelle sottoclassi
        return 0.0


class StandardBlocking(BlockingStrategy):
    """B1: Blocking basato su anno e marca"""
    
    def __init__(self, block_on: List[str] = None):
        """
        Args:
            block_on: Lista di attributi per il blocking
        """
        super().__init__('B1-Standard')
        self.block_on = block_on or ['year', 'manufacturer']
    
    def generate_pairs(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.MultiIndex:
        """
        Genera coppie candidate usando blocking standard
        
        Args:
            df1: Primo DataFrame
            df2: Secondo DataFrame
            
        Returns:
            MultiIndex con le coppie candidate
        """
        indexer = rl.Index()
        
        # Blocking su multiple colonne
        for col in self.block_on:
            if col in df1.columns and col in df2.columns:
                indexer.block(col)
        
        self.candidate_pairs = indexer.index(df1, df2)
        
        total_possible = len(df1) * len(df2)
        reduction = (1 - len(self.candidate_pairs) / total_possible) * 100
        
        print(f"{self.name} Blocking:")
        print(f"  Coppie totali possibili: {total_possible:,}")
        print(f"  Coppie candidate: {len(self.candidate_pairs):,}")
        print(f"  Riduzione: {reduction:.2f}%")
        
        return self.candidate_pairs


class ExtendedBlocking(BlockingStrategy):
    """B2: Blocking esteso con prefisso modello"""
    
    def __init__(self, block_on: List[str] = None, model_prefix_len: int = 3):
        """
        Args:
            block_on: Lista di attributi per il blocking
            model_prefix_len: Lunghezza del prefisso del modello
        """
        super().__init__('B2-Extended')
        self.block_on = block_on or ['year', 'manufacturer']
        self.model_prefix_len = model_prefix_len
    
    def _add_model_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge colonna con prefisso del modello
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame con colonna model_prefix
        """
        df = df.copy()
        
        if 'model' in df.columns:
            df['model_prefix'] = df['model'].fillna('').astype(str).str[:self.model_prefix_len].str.lower()
        else:
            df['model_prefix'] = ''
        
        return df
    
    def generate_pairs(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.MultiIndex:
        """
        Genera coppie candidate usando blocking esteso
        
        Args:
            df1: Primo DataFrame
            df2: Secondo DataFrame
            
        Returns:
            MultiIndex con le coppie candidate
        """
        # Aggiungi prefisso modello
        df1_with_prefix = self._add_model_prefix(df1)
        df2_with_prefix = self._add_model_prefix(df2)
        
        indexer = rl.Index()
        
        # Blocking su attributi standard
        for col in self.block_on:
            if col in df1_with_prefix.columns and col in df2_with_prefix.columns:
                indexer.block(col)
        
        # Blocking aggiuntivo su prefisso modello
        indexer.block('model_prefix')
        
        self.candidate_pairs = indexer.index(df1_with_prefix, df2_with_prefix)
        
        total_possible = len(df1) * len(df2)
        reduction = (1 - len(self.candidate_pairs) / total_possible) * 100
        
        print(f"{self.name} Blocking:")
        print(f"  Coppie totali possibili: {total_possible:,}")
        print(f"  Coppie candidate: {len(self.candidate_pairs):,}")
        print(f"  Riduzione: {reduction:.2f}%")
        
        return self.candidate_pairs


class SortedNeighborhoodBlocking(BlockingStrategy):
    """Blocking basato su Sorted Neighborhood"""
    
    def __init__(self, on: str = 'year', window: int = 5):
        """
        Args:
            on: Attributo per il sorting
            window: Dimensione della finestra
        """
        super().__init__('Sorted-Neighborhood')
        self.sort_on = on
        self.window = window
    
    def generate_pairs(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.MultiIndex:
        """
        Genera coppie candidate usando sorted neighborhood
        
        Args:
            df1: Primo DataFrame
            df2: Secondo DataFrame
            
        Returns:
            MultiIndex con le coppie candidate
        """
        indexer = rl.Index()
        
        if self.sort_on in df1.columns and self.sort_on in df2.columns:
            indexer.sortedneighbourhood(self.sort_on, window=self.window)
        
        self.candidate_pairs = indexer.index(df1, df2)
        
        total_possible = len(df1) * len(df2)
        reduction = (1 - len(self.candidate_pairs) / total_possible) * 100
        
        print(f"{self.name} Blocking:")
        print(f"  Coppie totali possibili: {total_possible:,}")
        print(f"  Coppie candidate: {len(self.candidate_pairs):,}")
        print(f"  Riduzione: {reduction:.2f}%")
        
        return self.candidate_pairs


class BlockingEvaluator:
    """Classe per valutare le strategie di blocking"""
    
    def __init__(self, ground_truth: pd.DataFrame):
        """
        Args:
            ground_truth: DataFrame con ground truth (con label)
        """
        self.ground_truth = ground_truth
    
    def evaluate_blocking(self, candidate_pairs: pd.MultiIndex, 
                         df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        """
        Valuta una strategia di blocking
        
        Args:
            candidate_pairs: Coppie candidate generate dal blocking
            df1: Primo DataFrame
            df2: Secondo DataFrame
            
        Returns:
            Dizionario con metriche di valutazione
        """
        # Crea set di coppie candidate
        candidate_set = set(candidate_pairs)
        
        # Crea set di coppie vere (ground truth positivi)
        true_pairs = self.ground_truth[self.ground_truth['label'] == 1]
        
        # Converti record_id a indici
        df1_id_to_idx = {row['record_id']: idx for idx, row in df1.iterrows()}
        df2_id_to_idx = {row['record_id']: idx for idx, row in df2.iterrows()}
        
        true_pair_set = set()
        for _, row in true_pairs.iterrows():
            idx1 = df1_id_to_idx.get(row['record_id_1'])
            idx2 = df2_id_to_idx.get(row['record_id_2'])
            if idx1 is not None and idx2 is not None:
                true_pair_set.add((idx1, idx2))
        
        # Calcola metriche
        true_positives = len(candidate_set & true_pair_set)
        false_negatives = len(true_pair_set - candidate_set)
        
        pair_completeness = true_positives / len(true_pair_set) if len(true_pair_set) > 0 else 0
        reduction_ratio = 1 - (len(candidate_pairs) / (len(df1) * len(df2)))
        
        metrics = {
            'total_candidate_pairs': len(candidate_pairs),
            'true_pairs_in_ground_truth': len(true_pair_set),
            'true_pairs_found': true_positives,
            'true_pairs_missed': false_negatives,
            'pair_completeness': pair_completeness,
            'reduction_ratio': reduction_ratio,
            'pairs_to_compare_ratio': len(candidate_pairs) / (len(df1) * len(df2))
        }
        
        return metrics
    
    def compare_strategies(self, strategies_results: dict) -> pd.DataFrame:
        """
        Confronta multiple strategie di blocking
        
        Args:
            strategies_results: Dizionario {nome_strategia: metriche}
            
        Returns:
            DataFrame con confronto
        """
        comparison_data = []
        
        for strategy_name, metrics in strategies_results.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Candidate Pairs': f"{metrics['total_candidate_pairs']:,}",
                'Pair Completeness': f"{metrics['pair_completeness']:.2%}",
                'Reduction Ratio': f"{metrics['reduction_ratio']:.2%}",
                'True Pairs Found': metrics['true_pairs_found'],
                'True Pairs Missed': metrics['true_pairs_missed']
            })
        
        return pd.DataFrame(comparison_data)
