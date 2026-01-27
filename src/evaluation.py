"""
Evaluation Module
Valuta le prestazioni delle diverse pipeline di record linkage
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import time


class PipelineEvaluator:
    """Classe per valutare le pipeline di record linkage"""
    
    def __init__(self, ground_truth: pd.DataFrame):
        """
        Args:
            ground_truth: DataFrame con ground truth
        """
        self.ground_truth = ground_truth
        self.results = {}
    
    def evaluate_predictions(self, predictions: pd.DataFrame,
                           pipeline_name: str,
                           df1: pd.DataFrame = None,
                           df2: pd.DataFrame = None) -> Dict:
        """
        Valuta le predizioni di una pipeline
        
        Args:
            predictions: DataFrame con predizioni (deve avere 'prediction' o 'label')
            pipeline_name: Nome della pipeline
            df1: Primo DataFrame (opzionale, per mapping)
            df2: Secondo DataFrame (opzionale, per mapping)
            
        Returns:
            Dizionario con metriche
        """
        # Prepara predizioni e ground truth per confronto
        y_true, y_pred = self._align_predictions(predictions, df1, df2)
        
        # Calcola metriche
        metrics = {
            'pipeline_name': pipeline_name,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'total_predictions': len(y_pred),
            'predicted_matches': int(y_pred.sum()),
            'true_matches': int(y_true.sum()),
            'true_positives': int(((y_true == 1) & (y_pred == 1)).sum()),
            'false_positives': int(((y_true == 0) & (y_pred == 1)).sum()),
            'false_negatives': int(((y_true == 1) & (y_pred == 0)).sum()),
            'true_negatives': int(((y_true == 0) & (y_pred == 0)).sum())
        }
        
        self.results[pipeline_name] = metrics
        
        print(f"\n{'='*60}")
        print(f"Risultati per {pipeline_name}")
        print(f"{'='*60}")
        print(f"Precision:  {metrics['precision']:.4f}")
        print(f"Recall:     {metrics['recall']:.4f}")
        print(f"F1-Score:   {metrics['f1_score']:.4f}")
        print(f"Accuracy:   {metrics['accuracy']:.4f}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def _align_predictions(self, predictions: pd.DataFrame,
                          df1: pd.DataFrame = None,
                          df2: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Allinea predizioni e ground truth
        
        Args:
            predictions: DataFrame con predizioni
            df1: Primo DataFrame
            df2: Secondo DataFrame
            
        Returns:
            Tuple (y_true, y_pred)
        """
        # Crea mapping tra record_id e predizioni
        if df1 is not None and df2 is not None:
            # Crea mapping indici -> record_id
            idx_to_id1 = {idx: row['record_id'] for idx, row in df1.iterrows() if 'record_id' in df1.columns}
            idx_to_id2 = {idx: row['record_id'] for idx, row in df2.iterrows() if 'record_id' in df2.columns}
            
            # Mappa predizioni a record_id
            pred_pairs = set()
            for _, row in predictions.iterrows():
                if 'index_1' in row and 'index_2' in row:
                    id1 = idx_to_id1.get(row['index_1'])
                    id2 = idx_to_id2.get(row['index_2'])
                    if id1 and id2:
                        label = row.get('prediction', row.get('label', 0))
                        pred_pairs.add((id1, id2, label))
        else:
            # Usa direttamente i record_id se disponibili
            pred_pairs = set()
            for _, row in predictions.iterrows():
                id1 = row.get('record_id_1')
                id2 = row.get('record_id_2')
                label = row.get('prediction', row.get('label', 0))
                if id1 and id2:
                    pred_pairs.add((id1, id2, label))
        
        # Allinea con ground truth
        y_true = []
        y_pred = []
        
        for _, gt_row in self.ground_truth.iterrows():
            id1 = gt_row['record_id_1']
            id2 = gt_row['record_id_2']
            true_label = gt_row['label']
            
            # Cerca la predizione corrispondente
            pred_label = 0  # Default: non-match
            for pred_id1, pred_id2, pred in pred_pairs:
                if (pred_id1 == id1 and pred_id2 == id2) or \
                   (pred_id1 == id2 and pred_id2 == id1):
                    pred_label = pred
                    break
            
            y_true.append(true_label)
            y_pred.append(pred_label)
        
        return np.array(y_true), np.array(y_pred)
    
    def add_timing_info(self, pipeline_name: str, 
                       training_time: float = None,
                       inference_time: float = None):
        """
        Aggiunge informazioni sui tempi
        
        Args:
            pipeline_name: Nome della pipeline
            training_time: Tempo di training in secondi
            inference_time: Tempo di inferenza in secondi
        """
        if pipeline_name in self.results:
            if training_time is not None:
                self.results[pipeline_name]['training_time'] = training_time
            if inference_time is not None:
                self.results[pipeline_name]['inference_time'] = inference_time
    
    def compare_pipelines(self) -> pd.DataFrame:
        """
        Confronta tutte le pipeline valutate
        
        Returns:
            DataFrame con confronto
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for pipeline_name, metrics in self.results.items():
            row = {
                'Pipeline': pipeline_name,
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Predictions': f"{metrics['total_predictions']:,}",
                'Matches Found': f"{metrics['predicted_matches']:,}"
            }
            
            if 'training_time' in metrics:
                row['Training Time (s)'] = f"{metrics['training_time']:.2f}"
            
            if 'inference_time' in metrics:
                row['Inference Time (s)'] = f"{metrics['inference_time']:.2f}"
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_metrics_comparison(self, save_path: str = None):
        """
        Visualizza confronto delle metriche
        
        Args:
            save_path: Percorso dove salvare il grafico
        """
        if not self.results:
            print("No results to plot")
            return
        
        # Prepara dati
        pipelines = list(self.results.keys())
        precision = [self.results[p]['precision'] for p in pipelines]
        recall = [self.results[p]['recall'] for p in pipelines]
        f1 = [self.results[p]['f1_score'] for p in pipelines]
        
        # Crea grafico
        x = np.arange(len(pipelines))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
        bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon')
        
        ax.set_xlabel('Pipeline', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Confronto Metriche per Pipeline', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pipelines, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Aggiungi valori sulle barre
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_timing_comparison(self, save_path: str = None):
        """
        Visualizza confronto dei tempi
        
        Args:
            save_path: Percorso dove salvare il grafico
        """
        # Filtra pipeline con informazioni sui tempi
        pipelines_with_timing = {k: v for k, v in self.results.items() 
                                if 'training_time' in v or 'inference_time' in v}
        
        if not pipelines_with_timing:
            print("No timing information available")
            return
        
        pipelines = list(pipelines_with_timing.keys())
        training_times = [pipelines_with_timing[p].get('training_time', 0) for p in pipelines]
        inference_times = [pipelines_with_timing[p].get('inference_time', 0) for p in pipelines]
        
        x = np.arange(len(pipelines))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, training_times, width, label='Training Time', color='steelblue')
        bars2 = ax.bar(x + width/2, inference_times, width, label='Inference Time', color='coral')
        
        ax.set_xlabel('Pipeline', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Confronto Tempi di Esecuzione', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pipelines, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, predictions: pd.DataFrame,
                             pipeline_name: str,
                             df1: pd.DataFrame = None,
                             df2: pd.DataFrame = None,
                             save_path: str = None):
        """
        Visualizza confusion matrix
        
        Args:
            predictions: DataFrame con predizioni
            pipeline_name: Nome della pipeline
            df1: Primo DataFrame
            df2: Secondo DataFrame
            save_path: Percorso dove salvare il grafico
        """
        y_true, y_pred = self._align_predictions(predictions, df1, df2)
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Match', 'Match'],
                   yticklabels=['Non-Match', 'Match'])
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix: {pipeline_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, output_file: str):
        """
        Genera report completo della valutazione
        
        Args:
            output_file: Percorso del file di output
        """
        report = f"""
{'='*80}
REPORT VALUTAZIONE PIPELINE DI RECORD LINKAGE
{'='*80}

Data di generazione: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Ground Truth: {len(self.ground_truth)} coppie
  - Match: {(self.ground_truth['label'] == 1).sum()}
  - Non-Match: {(self.ground_truth['label'] == 0).sum()}

{'='*80}
RISULTATI PER PIPELINE
{'='*80}

"""
        
        for pipeline_name, metrics in self.results.items():
            report += f"""
{'-'*80}
Pipeline: {pipeline_name}
{'-'*80}

Metriche di Performance:
  Precision:        {metrics['precision']:.4f}
  Recall:           {metrics['recall']:.4f}
  F1-Score:         {metrics['f1_score']:.4f}
  Accuracy:         {metrics['accuracy']:.4f}

Conteggi:
  Predizioni totali:    {metrics['total_predictions']:,}
  Match predetti:       {metrics['predicted_matches']:,}
  True Positives:       {metrics['true_positives']:,}
  False Positives:      {metrics['false_positives']:,}
  False Negatives:      {metrics['false_negatives']:,}
  True Negatives:       {metrics['true_negatives']:,}
"""
            
            if 'training_time' in metrics:
                report += f"\nTempo di Training:    {metrics['training_time']:.2f} secondi"
            
            if 'inference_time' in metrics:
                report += f"\nTempo di Inferenza:   {metrics['inference_time']:.2f} secondi"
            
            report += "\n"
        
        report += f"\n{'='*80}\n"
        report += "CONFRONTO PIPELINE\n"
        report += f"{'='*80}\n\n"
        report += self.compare_pipelines().to_string(index=False)
        report += f"\n\n{'='*80}\n"
        
        # Salva report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report salvato in {output_file}")
        
        return report
