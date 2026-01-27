"""
Schema Mediation Module
Definisce lo schema mediato e gestisce il mapping degli attributi
"""

import pandas as pd
from typing import Dict, List, Callable
import yaml


class MediatedSchema:
    """Classe per gestire lo schema mediato"""
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Percorso del file di configurazione
        """
        self.config_path = config_path
        self.schema = self._load_schema()
        self.attribute_mappings = {}
        self.transformation_functions = {}
    
    def _load_schema(self) -> List[str]:
        """
        Carica lo schema mediato dalla configurazione
        
        Returns:
            Lista di attributi dello schema mediato
        """
        if self.config_path:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('mediated_schema', {}).get('attributes', [])
        
        # Schema mediato di default
        return [
            'vin',
            'price',
            'year',
            'manufacturer',
            'model',
            'condition',
            'cylinders',
            'fuel',
            'odometer',
            'transmission',
            'drive',
            'size',
            'type',
            'paint_color',
            'state',
            'description'
        ]
    
    def add_source_mapping(self, source_name: str, mapping: Dict[str, str]):
        """
        Aggiunge un mapping per una sorgente
        
        Args:
            source_name: Nome della sorgente
            mapping: Dizionario {attributo_sorgente: attributo_mediato}
        """
        self.attribute_mappings[source_name] = mapping
    
    def add_transformation(self, attribute: str, func: Callable):
        """
        Aggiunge una funzione di trasformazione per un attributo
        
        Args:
            attribute: Nome dell'attributo
            func: Funzione di trasformazione
        """
        self.transformation_functions[attribute] = func
    
    def get_craigslist_mapping(self) -> Dict[str, str]:
        """
        Mapping per Craigslist dataset
        
        Returns:
            Dizionario di mapping
        """
        return {
            'VIN': 'vin',
            'price': 'price',
            'year': 'year',
            'manufacturer': 'manufacturer',
            'model': 'model',
            'condition': 'condition',
            'cylinders': 'cylinders',
            'fuel': 'fuel',
            'odometer': 'odometer',
            'transmission': 'transmission',
            'drive': 'drive',
            'size': 'size',
            'type': 'type',
            'paint_color': 'paint_color',
            'state': 'state',
            'description': 'description'
        }
    
    def get_usedcars_mapping(self) -> Dict[str, str]:
        """
        Mapping per US Used Cars dataset
        
        Returns:
            Dizionario di mapping
        """
        # Mapping aggiornato con i nomi corretti delle colonne
        return {
            'vin': 'vin',
            'price': 'price',
            'year': 'year',
            'make_name': 'manufacturer',
            'model_name': 'model',
            'body_type': 'type',
            'fuel_type': 'fuel',
            'transmission': 'transmission',
            'mileage': 'odometer',
            'exterior_color': 'paint_color',
            'engine_cylinders': 'cylinders',
            'wheel_system': 'drive',
            'description': 'description'
        }
    
    def normalize_manufacturer(self, manufacturer: str) -> str:
        """
        Normalizza i nomi dei produttori
        
        Args:
            manufacturer: Nome del produttore
            
        Returns:
            Nome normalizzato
        """
        if pd.isna(manufacturer):
            return None
        
        manufacturer = str(manufacturer).lower().strip()
        
        # Mapping comuni
        mappings = {
            'chevy': 'chevrolet',
            'vw': 'volkswagen',
            'mercedes': 'mercedes-benz',
            'mercedesbenz': 'mercedes-benz',
            'benz': 'mercedes-benz'
        }
        
        return mappings.get(manufacturer, manufacturer)
    
    def normalize_fuel_type(self, fuel: str) -> str:
        """
        Normalizza i tipi di carburante
        
        Args:
            fuel: Tipo di carburante
            
        Returns:
            Tipo normalizzato
        """
        if pd.isna(fuel):
            return None
        
        fuel = str(fuel).lower().strip()
        
        mappings = {
            'gasoline': 'gas',
            'petrol': 'gas',
            'electric': 'electric',
            'ev': 'electric',
            'hybrid': 'hybrid',
            'diesel': 'diesel'
        }
        
        return mappings.get(fuel, fuel)
    
    def normalize_transmission(self, transmission: str) -> str:
        """
        Normalizza i tipi di trasmissione
        
        Args:
            transmission: Tipo di trasmissione
            
        Returns:
            Tipo normalizzato
        """
        if pd.isna(transmission):
            return None
        
        transmission = str(transmission).lower().strip()
        
        if 'automatic' in transmission or 'auto' in transmission:
            return 'automatic'
        elif 'manual' in transmission:
            return 'manual'
        else:
            return 'other'
    
    def convert_price(self, price: any) -> float:
        """
        Converte il prezzo in formato numerico
        
        Args:
            price: Prezzo in vari formati
            
        Returns:
            Prezzo come float
        """
        if pd.isna(price):
            return None
        
        try:
            # Rimuovi simboli di valuta e virgole
            price_str = str(price).replace('$', '').replace(',', '').strip()
            price_float = float(price_str)
            
            # Filtra valori non realistici
            if price_float < 100 or price_float > 500000:
                return None
            
            return price_float
        except:
            return None
    
    def convert_year(self, year: any) -> int:
        """
        Converte l'anno in formato intero
        
        Args:
            year: Anno in vari formati
            
        Returns:
            Anno come int
        """
        if pd.isna(year):
            return None
        
        try:
            year_int = int(float(year))
            
            # Filtra valori non realistici (auto dal 1900 al futuro prossimo)
            if year_int < 1900 or year_int > 2026:
                return None
            
            return year_int
        except:
            return None
    
    def convert_odometer(self, odometer: any) -> float:
        """
        Converte il chilometraggio in formato numerico
        
        Args:
            odometer: Chilometraggio in vari formati
            
        Returns:
            Chilometraggio come float
        """
        if pd.isna(odometer):
            return None
        
        try:
            odometer_str = str(odometer).replace(',', '').strip()
            odometer_float = float(odometer_str)
            
            # Filtra valori non realistici
            if odometer_float < 0 or odometer_float > 1000000:
                return None
            
            return odometer_float
        except:
            return None
    
    def clean_vin(self, vin: str) -> str:
        """
        Pulisce e normalizza il VIN
        
        Args:
            vin: VIN da pulire
            
        Returns:
            VIN pulito
        """
        if pd.isna(vin):
            return None
        
        # Converti in maiuscolo e rimuovi spazi
        vin = str(vin).upper().strip()
        
        # Rimuovi caratteri non alfanumerici
        vin = ''.join(c for c in vin if c.isalnum())
        
        # VIN standard è di 17 caratteri, ma accettiamo anche VIN parziali
        if len(vin) < 10:  # VIN troppo corto probabilmente non è valido
            return None
        
        return vin


class SourceAligner:
    """Classe per allineare le sorgenti allo schema mediato"""
    
    def __init__(self, mediated_schema: MediatedSchema):
        """
        Args:
            mediated_schema: Schema mediato di riferimento
        """
        self.schema = mediated_schema
    
    def align_craigslist(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Allinea il dataset Craigslist allo schema mediato
        
        Args:
            df: DataFrame Craigslist
            
        Returns:
            DataFrame allineato
        """
        mapping = self.schema.get_craigslist_mapping()
        
        # Crea nuovo DataFrame con schema mediato
        aligned_df = pd.DataFrame()
        
        for source_col, target_col in mapping.items():
            if source_col in df.columns:
                aligned_df[target_col] = df[source_col]
        
        # Applica trasformazioni
        aligned_df = self._apply_transformations(aligned_df)
        
        # Aggiungi colonna sorgente
        aligned_df['source'] = 'craigslist'
        
        return aligned_df
    
    def align_usedcars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Allinea il dataset US Used Cars allo schema mediato
        
        Args:
            df: DataFrame US Used Cars
            
        Returns:
            DataFrame allineato
        """
        mapping = self.schema.get_usedcars_mapping()
        
        # Crea nuovo DataFrame con schema mediato
        aligned_df = pd.DataFrame()
        
        for source_col, target_col in mapping.items():
            if source_col in df.columns:
                aligned_df[target_col] = df[source_col]
        
        # Applica trasformazioni
        aligned_df = self._apply_transformations(aligned_df)
        
        # Aggiungi colonna sorgente
        aligned_df['source'] = 'usedcars'
        
        return aligned_df
    
    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applica trasformazioni agli attributi
        
        Args:
            df: DataFrame da trasformare
            
        Returns:
            DataFrame trasformato
        """
        # Applica normalizzazioni
        if 'manufacturer' in df.columns:
            df['manufacturer'] = df['manufacturer'].apply(self.schema.normalize_manufacturer)
        
        if 'fuel' in df.columns:
            df['fuel'] = df['fuel'].apply(self.schema.normalize_fuel_type)
        
        if 'transmission' in df.columns:
            df['transmission'] = df['transmission'].apply(self.schema.normalize_transmission)
        
        if 'price' in df.columns:
            df['price'] = df['price'].apply(self.schema.convert_price)
        
        if 'year' in df.columns:
            df['year'] = df['year'].apply(self.schema.convert_year)
        
        if 'odometer' in df.columns:
            df['odometer'] = df['odometer'].apply(self.schema.convert_odometer)
        
        if 'vin' in df.columns:
            df['vin'] = df['vin'].apply(self.schema.clean_vin)
        
        return df
