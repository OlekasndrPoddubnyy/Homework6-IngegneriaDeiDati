"""
Record Linkage Module
Implementa regole di record linkage usando Python Record Linkage
"""

import pandas as pd
import recordlinkage as rl
from recordlinkage.base import BaseCompareFeature
import numpy as np
from typing import Dict, List


class RecordLinkageClassifier:
    """Classifier basato su regole per record linkage"""
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Configurazione per il comparison
        """
        self.config = config or self._default_config()
        self.compare = rl.Compare()
        self.features = None
        self.matches = None
    
    def _default_config(self) -> Dict:
        """
        Configurazione di default per il comparison
        
        Returns:
            Dizionario di configurazione
        """
        return {
            'comparison_fields': {
                'year': {
                    'method': 'exact',
                    'weight': 0.15
                },
                'manufacturer': {
                    'method': 'string',
                    'threshold': 0.85,
                    'weight': 0.20
                },
                'model': {
                    'method': 'string',
                    'threshold': 0.85,
                    'weight': 0.20
                },
                'price': {
                    'method': 'numeric',
                    'threshold': 0.10,
                    'weight': 0.10
                },
                'odometer': {
                    'method': 'numeric',
                    'threshold': 0.15,
                    'weight': 0.15
                },
                'fuel': {
                    'method': 'exact',
                    'weight': 0.05
                },
                'transmission': {
                    'method': 'exact',
                    'weight': 0.05
                },
                'type': {
                    'method': 'string',
                    'threshold': 0.80,
                    'weight': 0.10
                }
            },
            'match_threshold': 0.7,
            'possible_match_threshold': 0.5
        }
    
    def setup_comparisons(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """
        Configura le funzioni di comparison
        
        Args:
            df1: Primo DataFrame
            df2: Secondo DataFrame
        """
        comparison_fields = self.config['comparison_fields']
        
        for field, params in comparison_fields.items():
            if field not in df1.columns or field not in df2.columns:
                print(f"Warning: Field '{field}' not found in both DataFrames, skipping")
                continue
            
            method = params['method']
            
            if method == 'exact':
                self.compare.exact(field, field, label=field)
            
            elif method == 'string':
                threshold = params.get('threshold', 0.85)
                # Usa Jaro-Winkler per similarità stringhe
                self.compare.string(field, field, method='jarowinkler', 
                                   threshold=threshold, label=field)
            
            elif method == 'numeric':
                threshold = params.get('threshold', 0.10)
                # Usa comparison numerico con threshold percentuale
                self.compare.numeric(field, field, method='gauss', 
                                    offset=threshold, label=field)
    
    def compute_features(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                        candidate_pairs: pd.MultiIndex) -> pd.DataFrame:
        """
        Calcola feature di similarità per le coppie candidate
        
        Args:
            df1: Primo DataFrame
            df2: Secondo DataFrame
            candidate_pairs: Coppie candidate dal blocking
            
        Returns:
            DataFrame con feature di similarità
        """
        self.setup_comparisons(df1, df2)
        self.features = self.compare.compute(candidate_pairs, df1, df2)
        
        print(f"Feature calcolate per {len(self.features)} coppie")
        
        return self.features
    
    def compute_weighted_score(self, features: pd.DataFrame = None) -> pd.Series:
        """
        Calcola uno score pesato per ogni coppia
        
        Args:
            features: DataFrame con feature (usa self.features se None)
            
        Returns:
            Series con gli score
        """
        if features is None:
            features = self.features
        
        if features is None:
            raise ValueError("No features available. Run compute_features first.")
        
        # Calcola score pesato
        comparison_fields = self.config['comparison_fields']
        scores = pd.Series(0.0, index=features.index)
        
        for field, params in comparison_fields.items():
            if field in features.columns:
                weight = params['weight']
                # Get the field as a Series and ensure it's numeric
                field_data = features[field]
                
                # If field_data is a DataFrame (duplicate columns), take the first column
                if isinstance(field_data, pd.DataFrame):
                    field_data = field_data.iloc[:, 0]
                
                # Convert to Series if needed and make numeric
                if not isinstance(field_data, pd.Series):
                    field_data = pd.Series(field_data, index=features.index)
                
                # Convert to numeric, coercing errors to NaN, then fill NaN with 0
                field_data = pd.to_numeric(field_data, errors='coerce').fillna(0)
                
                # Multiply by weight and add to scores
                field_scores = field_data * weight
                scores = scores.add(field_scores, fill_value=0)
        
        return scores
    
    def classify(self, df1: pd.DataFrame, df2: pd.DataFrame,
                candidate_pairs: pd.MultiIndex) -> pd.DataFrame:
        """
        Classifica le coppie come match/non-match
        
        Args:
            df1: Primo DataFrame
            df2: Secondo DataFrame
            candidate_pairs: Coppie candidate dal blocking
            
        Returns:
            DataFrame con classificazione
        """
        # Calcola feature
        features = self.compute_features(df1, df2, candidate_pairs)
        
        # Calcola score
        scores = self.compute_weighted_score(features)
        
        # Classifica basandosi su threshold
        match_threshold = self.config['match_threshold']
        
        results = pd.DataFrame({
            'score': scores,
            'prediction': (scores >= match_threshold).astype(int)
        })
        
        self.matches = results[results['prediction'] == 1]
        
        print(f"Match trovati: {len(self.matches)}")
        print(f"Score medio match: {self.matches['score'].mean():.3f}")
        
        return results
    
    def get_matches_with_details(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                 top_n: int = None) -> pd.DataFrame:
        """
        Ottiene i match con dettagli dei record
        
        Args:
            df1: Primo DataFrame
            df2: Secondo DataFrame
            top_n: Numero massimo di match da ritornare
            
        Returns:
            DataFrame con dettagli dei match
        """
        if self.matches is None:
            raise ValueError("No matches available. Run classify first.")
        
        matches = self.matches.copy()
        
        if top_n:
            matches = matches.nlargest(top_n, 'score')
        
        # Aggiungi dettagli dei record
        match_details = []
        
        for (idx1, idx2), row in matches.iterrows():
            details = {
                'index_1': idx1,
                'index_2': idx2,
                'score': row['score']
            }
            
            # Aggiungi campi dal primo record
            if idx1 in df1.index:
                for col in ['year', 'manufacturer', 'model', 'price', 'odometer']:
                    if col in df1.columns:
                        details[f'{col}_1'] = df1.loc[idx1, col]
            
            # Aggiungi campi dal secondo record
            if idx2 in df2.index:
                for col in ['year', 'manufacturer', 'model', 'price', 'odometer']:
                    if col in df2.columns:
                        details[f'{col}_2'] = df2.loc[idx2, col]
            
            match_details.append(details)
        
        return pd.DataFrame(match_details)


class FeatureEngineering:
    """Classe per feature engineering avanzato"""
    
    @staticmethod
    def add_price_difference(features: pd.DataFrame, df1: pd.DataFrame, 
                            df2: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge feature basata sulla differenza di prezzo
        
        Args:
            features: DataFrame con feature esistenti
            df1: Primo DataFrame
            df2: Secondo DataFrame
            
        Returns:
            DataFrame con feature aggiuntive
        """
        if 'price' not in df1.columns or 'price' not in df2.columns:
            return features
        
        features = features.copy()
        
        # Calcola differenza percentuale di prezzo
        price_diff = []
        for idx1, idx2 in features.index:
            p1 = df1.loc[idx1, 'price']
            p2 = df2.loc[idx2, 'price']
            
            if pd.notna(p1) and pd.notna(p2) and p1 > 0:
                diff = abs(p1 - p2) / p1
                price_diff.append(1 - min(diff, 1.0))
            else:
                price_diff.append(0.0)
        
        features['price_similarity'] = price_diff
        
        return features
    
    @staticmethod
    def add_year_difference(features: pd.DataFrame, df1: pd.DataFrame,
                           df2: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge feature basata sulla differenza di anno
        
        Args:
            features: DataFrame con feature esistenti
            df1: Primo DataFrame
            df2: Secondo DataFrame
            
        Returns:
            DataFrame con feature aggiuntive
        """
        if 'year' not in df1.columns or 'year' not in df2.columns:
            return features
        
        features = features.copy()
        
        # Calcola differenza assoluta di anno
        year_diff = []
        for idx1, idx2 in features.index:
            y1 = df1.loc[idx1, 'year']
            y2 = df2.loc[idx2, 'year']
            
            if pd.notna(y1) and pd.notna(y2):
                diff = abs(y1 - y2)
                # Converti in similarità (0 diff = 1.0, 5+ diff = 0.0)
                similarity = max(1 - diff / 5, 0)
                year_diff.append(similarity)
            else:
                year_diff.append(0.0)
        
        features['year_similarity'] = year_diff
        
        return features
    
    @staticmethod
    def add_aggregate_features(features: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge feature aggregate
        
        Args:
            features: DataFrame con feature esistenti
            
        Returns:
            DataFrame con feature aggiuntive
        """
        features = features.copy()
        
        # Somma di tutte le similarità
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features['total_similarity'] = features[numeric_cols].sum(axis=1)
        
        # Media di tutte le similarità
        features['avg_similarity'] = features[numeric_cols].mean(axis=1)
        
        # Numero di match esatti
        features['exact_matches'] = (features[numeric_cols] == 1.0).sum(axis=1)
        
        return features
