"""
Data Analysis Module
Analizza la percentuale di valori nulli e unici per ogni attributo
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class DataAnalyzer:
    """Classe per analizzare la qualità dei dati"""
    
    def __init__(self, df: pd.DataFrame, source_name: str):
        """
        Args:
            df: DataFrame da analizzare
            source_name: Nome della sorgente dati
        """
        self.df = df
        self.source_name = source_name
        self.analysis_results = {}
    
    def analyze_null_values(self) -> pd.DataFrame:
        """
        Analizza la percentuale di valori nulli per ogni attributo
        
        Returns:
            DataFrame con statistiche sui valori nulli
        """
        null_stats = pd.DataFrame({
            'attribute': self.df.columns,
            'null_count': self.df.isnull().sum().values,
            'null_percentage': (self.df.isnull().sum() / len(self.df) * 100).values,
            'non_null_count': self.df.notnull().sum().values
        })
        
        null_stats = null_stats.sort_values('null_percentage', ascending=False)
        self.analysis_results['null_values'] = null_stats
        
        return null_stats
    
    def analyze_unique_values(self) -> pd.DataFrame:
        """
        Analizza la percentuale di valori unici per ogni attributo
        
        Returns:
            DataFrame con statistiche sui valori unici
        """
        unique_stats = pd.DataFrame({
            'attribute': self.df.columns,
            'unique_count': [self.df[col].nunique() for col in self.df.columns],
            'unique_percentage': [(self.df[col].nunique() / len(self.df) * 100) 
                                  for col in self.df.columns],
            'total_count': [len(self.df)] * len(self.df.columns)
        })
        
        unique_stats = unique_stats.sort_values('unique_percentage', ascending=False)
        self.analysis_results['unique_values'] = unique_stats
        
        return unique_stats
    
    def analyze_data_types(self) -> pd.DataFrame:
        """
        Analizza i tipi di dati per ogni attributo
        
        Returns:
            DataFrame con informazioni sui tipi di dati
        """
        dtype_stats = pd.DataFrame({
            'attribute': self.df.columns,
            'dtype': [str(dtype) for dtype in self.df.dtypes.values],
            'memory_usage': self.df.memory_usage(deep=True).values[1:]  # Skip index
        })
        
        self.analysis_results['data_types'] = dtype_stats
        
        return dtype_stats
    
    def get_summary_statistics(self) -> Dict:
        """
        Genera statistiche riassuntive del dataset
        
        Returns:
            Dizionario con statistiche riassuntive
        """
        summary = {
            'source_name': self.source_name,
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'total_cells': len(self.df) * len(self.df.columns),
            'total_null_cells': self.df.isnull().sum().sum(),
            'null_percentage': (self.df.isnull().sum().sum() / 
                               (len(self.df) * len(self.df.columns)) * 100),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024 * 1024),
            'duplicate_rows': self.df.duplicated().sum()
        }
        
        self.analysis_results['summary'] = summary
        
        return summary
    
    def analyze_vin_column(self, vin_column: str = 'vin') -> Dict:
        """
        Analizza specificamente la colonna VIN
        
        Args:
            vin_column: Nome della colonna VIN
            
        Returns:
            Dizionario con statistiche sul VIN
        """
        if vin_column not in self.df.columns:
            return {'error': f'Column {vin_column} not found'}
        
        vin_data = self.df[vin_column].dropna()
        
        vin_stats = {
            'total_vins': len(vin_data),
            'unique_vins': vin_data.nunique(),
            'duplicate_vins': len(vin_data) - vin_data.nunique(),
            'null_vins': self.df[vin_column].isnull().sum(),
            'vin_lengths': vin_data.str.len().value_counts().to_dict(),
            'standard_vin_count': (vin_data.str.len() == 17).sum(),  # VIN standard è 17 caratteri
            'standard_vin_percentage': (vin_data.str.len() == 17).sum() / len(vin_data) * 100
        }
        
        self.analysis_results['vin_analysis'] = vin_stats
        
        return vin_stats
    
    def plot_null_values(self, top_n: int = 20, save_path: str = None):
        """
        Visualizza la percentuale di valori nulli
        
        Args:
            top_n: Numero di attributi da visualizzare
            save_path: Percorso dove salvare il grafico
        """
        if 'null_values' not in self.analysis_results:
            self.analyze_null_values()
        
        null_stats = self.analysis_results['null_values'].head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=null_stats, y='attribute', x='null_percentage', palette='RdYlGn_r')
        plt.xlabel('Percentuale Valori Nulli (%)', fontsize=12)
        plt.ylabel('Attributo', fontsize=12)
        plt.title(f'{self.source_name}: Percentuale Valori Nulli per Attributo (Top {top_n})', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_unique_values(self, top_n: int = 20, save_path: str = None):
        """
        Visualizza la percentuale di valori unici
        
        Args:
            top_n: Numero di attributi da visualizzare
            save_path: Percorso dove salvare il grafico
        """
        if 'unique_values' not in self.analysis_results:
            self.analyze_unique_values()
        
        unique_stats = self.analysis_results['unique_values'].head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=unique_stats, y='attribute', x='unique_percentage', palette='viridis')
        plt.xlabel('Percentuale Valori Unici (%)', fontsize=12)
        plt.ylabel('Attributo', fontsize=12)
        plt.title(f'{self.source_name}: Percentuale Valori Unici per Attributo (Top {top_n})', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_full_report(self, output_path: str = None) -> str:
        """
        Genera un report completo dell'analisi
        
        Args:
            output_path: Percorso dove salvare il report
            
        Returns:
            Report in formato stringa
        """
        # Esegui tutte le analisi
        self.analyze_null_values()
        self.analyze_unique_values()
        self.analyze_data_types()
        summary = self.get_summary_statistics()
        
        # Genera report
        report = f"""
{'='*80}
REPORT ANALISI DATI: {self.source_name}
{'='*80}

STATISTICHE RIASSUNTIVE
{'-'*80}
Numero totale di righe: {summary['total_rows']:,}
Numero totale di colonne: {summary['total_columns']}
Celle totali: {summary['total_cells']:,}
Celle nulle: {summary['total_null_cells']:,} ({summary['null_percentage']:.2f}%)
Uso memoria: {summary['memory_usage_mb']:.2f} MB
Righe duplicate: {summary['duplicate_rows']:,}

VALORI NULLI (Top 10)
{'-'*80}
{self.analysis_results['null_values'].head(10).to_string()}

VALORI UNICI (Top 10)
{'-'*80}
{self.analysis_results['unique_values'].head(10).to_string()}

TIPI DI DATI
{'-'*80}
{self.analysis_results['data_types'].to_string()}

{'='*80}
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


def compare_sources(analyzer1: DataAnalyzer, analyzer2: DataAnalyzer) -> pd.DataFrame:
    """
    Confronta due sorgenti di dati
    
    Args:
        analyzer1: Primo DataAnalyzer
        analyzer2: Secondo DataAnalyzer
        
    Returns:
        DataFrame con il confronto
    """
    summary1 = analyzer1.get_summary_statistics()
    summary2 = analyzer2.get_summary_statistics()
    
    comparison = pd.DataFrame({
        'Metrica': [
            'Righe',
            'Colonne',
            'Celle Totali',
            'Celle Nulle',
            'Percentuale Nulli',
            'Memoria (MB)',
            'Righe Duplicate'
        ],
        summary1['source_name']: [
            f"{summary1['total_rows']:,}",
            f"{summary1['total_columns']}",
            f"{summary1['total_cells']:,}",
            f"{summary1['total_null_cells']:,}",
            f"{summary1['null_percentage']:.2f}%",
            f"{summary1['memory_usage_mb']:.2f}",
            f"{summary1['duplicate_rows']:,}"
        ],
        summary2['source_name']: [
            f"{summary2['total_rows']:,}",
            f"{summary2['total_columns']}",
            f"{summary2['total_cells']:,}",
            f"{summary2['total_null_cells']:,}",
            f"{summary2['null_percentage']:.2f}%",
            f"{summary2['memory_usage_mb']:.2f}",
            f"{summary2['duplicate_rows']:,}"
        ]
    })
    
    return comparison
