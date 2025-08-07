#!/usr/bin/env python3
"""
Script de treino otimizado para datasets limpos
Usa datasets pequenos e eficientes para treino rápido
"""

import logging
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

class CleanDatasetTrainer:
    """Treinador para datasets limpos e otimizados"""
    
    def __init__(self, datasets_dir=None):
        if datasets_dir is None:
            self.datasets_dir = Path(__file__).parent.parent.parent / 'src' / 'datasets' / 'clean'
        else:
            self.datasets_dir = Path(datasets_dir)
        
        self.models_dir = Path(__file__).parent.parent.parent / 'src' / 'models' / 'clean'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.results = {}
    
    def load_dataset(self, dataset_name):
        """Carregar dataset limpo"""
        logger.info(f"Carregando dataset {dataset_name}...")
        
        try:
            # Carregar arrays
            X_train = np.load(self.datasets_dir / f'X_train_{dataset_name}.npy')
            X_test = np.load(self.datasets_dir / f'X_test_{dataset_name}.npy')
            y_train = np.load(self.datasets_dir / f'y_train_binary_{dataset_name}.npy')
            y_test = np.load(self.datasets_dir / f'y_test_binary_{dataset_name}.npy')
            
            # Carregar metadados
            with open(self.datasets_dir / f'metadata_{dataset_name}.json', 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Dataset {dataset_name} carregado:")
            logger.info(f"  Treino: {X_train.shape}")
            logger.info(f"  Teste: {X_test.shape}")
            logger.info(f"  Features: {X_train.shape[1]}")
            logger.info(f"  Ratio ataques: {y_train.mean():.2%}")
            
            return X_train, X_test, y_train, y_test, metadata
            
        except Exception as e:
            logger.error(f"Erro ao carregar dataset {dataset_name}: {e}")
            return None, None, None, None, None
    
    def train_random_forest(self, X_train, y_train, dataset_name):
        """Treinar Random Forest"""
        logger.info("Treinando Random Forest...")
        
        start_time = time.time()
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        logger.info(f"Random Forest treinado em {train_time:.2f}s")
        
        return rf, train_time
    
    def train_isolation_forest(self, X_train, dataset_name):
        """Treinar Isolation Forest"""
        logger.info("Treinando Isolation Forest...")
        
        start_time = time.time()
        
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        iso_forest.fit(X_train)
        
        train_time = time.time() - start_time
        logger.info(f"Isolation Forest treinado em {train_time:.2f}s")
        
        return iso_forest, train_time
    
    def train_logistic_regression(self, X_train, y_train, dataset_name):
        """Treinar Logistic Regression"""
        logger.info("Treinando Logistic Regression...")
        
        start_time = time.time()
        
        lr = LogisticRegression(
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        )
        
        lr.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        logger.info(f"Logistic Regression treinado em {train_time:.2f}s")
        
        return lr, train_time
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Avaliar modelo"""
        logger.info(f"Avaliando {model_name}...")
        
        start_time = time.time()
        
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):  # Isolation Forest
            scores = model.decision_function(X_test)
            y_pred = (scores < 0).astype(int)  # Anomalias como ataques
            y_proba = None
        else:
            y_pred = model.predict(X_test)
            y_proba = None
        
        eval_time = time.time() - start_time
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        logger.info(f"{model_name} - Acurácia: {accuracy:.4f}")
        logger.info(f"{model_name} - Precisão (Ataque): {report.get('1', {}).get('precision', 0):.4f}")
        logger.info(f"{model_name} - Recall (Ataque): {report.get('1', {}).get('recall', 0):.4f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_proba,
            'evaluation_time': eval_time
        }
    
    def train_dataset(self, dataset_name):
        """Treinar todos os modelos para um dataset"""
        logger.info(f"Iniciando treino para dataset {dataset_name}")
        
        # Carregar dados
        X_train, X_test, y_train, y_test, metadata = self.load_dataset(dataset_name)
        
        if X_train is None:
            logger.error(f"Falha ao carregar dataset {dataset_name}")
            return False
        
        dataset_results = {
            'metadata': metadata,
            'models': {}
        }
        
        # Treinar modelos
        models_to_train = [
            ('random_forest', self.train_random_forest),
            ('isolation_forest', self.train_isolation_forest),
            ('logistic_regression', self.train_logistic_regression)
        ]
        
        for model_name, train_func in models_to_train:
            try:
                logger.info(f"Treinando {model_name} para {dataset_name}...")
                
                if model_name == 'isolation_forest':
                    model, train_time = train_func(X_train, dataset_name)
                else:
                    model, train_time = train_func(X_train, y_train, dataset_name)
                
                # Avaliar modelo
                eval_results = self.evaluate_model(model, X_test, y_test, model_name)
                
                # Salvar modelo
                model_file = self.models_dir / f'{model_name}_{dataset_name}.pkl'
                joblib.dump(model, model_file)
                
                # Armazenar resultados
                dataset_results['models'][model_name] = {
                    'train_time': train_time,
                    'model_file': str(model_file),
                    **eval_results
                }
                
                logger.info(f"{model_name} para {dataset_name} concluído")
                
            except Exception as e:
                logger.error(f"Erro ao treinar {model_name} para {dataset_name}: {e}")
                continue
        
        # Salvar resultados
        results_file = self.models_dir / f'results_{dataset_name}.json'
        with open(results_file, 'w') as f:
            # Converter numpy arrays para listas para JSON
            clean_results = {}
            for key, value in dataset_results.items():
                if key == 'models':
                    clean_results[key] = {}
                    for model_name, model_results in value.items():
                        clean_model_results = {}
                        for k, v in model_results.items():
                            if isinstance(v, np.ndarray):
                                clean_model_results[k] = v.tolist()
                            else:
                                clean_model_results[k] = v
                        clean_results[key][model_name] = clean_model_results
                else:
                    clean_results[key] = value
            
            json.dump(clean_results, f, indent=2)
        
        self.results[dataset_name] = dataset_results
        logger.info(f"Treino completo para {dataset_name}")
        
        return True
    
    def generate_summary(self):
        """Gerar resumo dos resultados"""
        logger.info("Gerando resumo dos resultados...")
        
        summary = {
            'datasets_trained': list(self.results.keys()),
            'best_models': {},
            'training_summary': {}
        }
        
        for dataset_name, results in self.results.items():
            best_accuracy = 0
            best_model = None
            
            dataset_summary = {
                'total_models': len(results['models']),
                'models_performance': {}
            }
            
            for model_name, model_results in results['models'].items():
                accuracy = model_results['accuracy']
                dataset_summary['models_performance'][model_name] = {
                    'accuracy': accuracy,
                    'train_time': model_results['train_time']
                }
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
            
            summary['best_models'][dataset_name] = {
                'model': best_model,
                'accuracy': best_accuracy
            }
            summary['training_summary'][dataset_name] = dataset_summary
        
        # Salvar resumo
        summary_file = self.models_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Resumo salvo em training_summary.json")
        
        return summary

def main():
    """Função principal"""
    logger.info("Iniciando treino com datasets limpos...")
    
    # Verificar se datasets limpos existem
    datasets_dir = Path(__file__).parent.parent.parent / 'src' / 'datasets' / 'clean'
    
    if not datasets_dir.exists():
        logger.error("Datasets limpos não encontrados. Execute create_clean_datasets.py primeiro.")
        return False
    
    # Verificar datasets disponíveis
    available_datasets = []
    for dataset in ['cicddos', 'unsw', 'integrated']:
        if (datasets_dir / f'X_train_{dataset}.npy').exists():
            available_datasets.append(dataset)
    
    if not available_datasets:
        logger.error("Nenhum dataset limpo encontrado.")
        return False
    
    logger.info(f"Datasets disponíveis: {available_datasets}")
    
    # Inicializar treinador
    trainer = CleanDatasetTrainer(datasets_dir)
    
    # Treinar cada dataset
    success_count = 0
    for dataset_name in available_datasets:
        if trainer.train_dataset(dataset_name):
            success_count += 1
    
    if success_count == 0:
        logger.error("Nenhum dataset foi treinado com sucesso.")
        return False
    
    # Gerar resumo
    summary = trainer.generate_summary()
    
    logger.info(f"Treino concluído! {success_count}/{len(available_datasets)} datasets treinados.")
    logger.info("Resultados:")
    
    for dataset_name, best_info in summary['best_models'].items():
        logger.info(f"  {dataset_name}: {best_info['model']} (acurácia: {best_info['accuracy']:.4f})")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
