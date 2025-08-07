#!/usr/bin/env python3
"""
Script de treinamento otimizado para datasets realistas
Usa dados com overlap intencional para accuracy ~99.x%
"""

import logging
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def load_realistic_dataset(dataset_name, data_dir):
    """Carregar dataset realista"""
    logger.info(f"Carregando dataset realista: {dataset_name}")
    
    data_path = Path(data_dir)
    
    # Verificar se existe
    train_file = data_path / f'X_train_{dataset_name}.npy'
    if not train_file.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} não encontrado em {data_path}")
    
    # Carregar dados
    X_train = np.load(data_path / f'X_train_{dataset_name}.npy')
    X_test = np.load(data_path / f'X_test_{dataset_name}.npy')
    y_train = np.load(data_path / f'y_train_binary_{dataset_name}.npy')
    y_test = np.load(data_path / f'y_test_binary_{dataset_name}.npy')
    
    # Carregar metadados
    with open(data_path / f'metadata_{dataset_name}.json', 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Dataset carregado: {dataset_name}")
    logger.info(f"  Treino: {X_train.shape}")
    logger.info(f"  Teste: {X_test.shape}")
    logger.info(f"  Distribuição ataques: {metadata.get('attack_ratio', 'N/A'):.1%}")
    
    return X_train, X_test, y_train, y_test, metadata

def train_realistic_models(X_train, X_test, y_train, y_test, dataset_name):
    """Treinar modelos nos dados realistas"""
    logger.info(f"Treinando modelos para {dataset_name}...")
    
    models = {}
    results = {}
    
    # 1. Random Forest (otimizado para dados realistas)
    logger.info("Treinando Random Forest...")
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=100,      # Reduzido para evitar overfitting
        max_depth=15,          # Limitado para generalização
        min_samples_split=5,   # Aumentado para evitar overfitting
        min_samples_leaf=3,    # Aumentado para generalização
        max_features='sqrt',   # Reduzido conjunto de features
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    
    rf_time = time.time() - start_time
    
    # Cross-validation para validação
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
    
    results['random_forest'] = {
        'accuracy': float(accuracy_score(y_test, rf_pred)),
        'f1_score': float(f1_score(y_test, rf_pred)),
        'precision': float(precision_score(y_test, rf_pred)),
        'recall': float(recall_score(y_test, rf_pred)),
        'training_time': rf_time,
        'cv_f1_mean': float(cv_scores.mean()),
        'cv_f1_std': float(cv_scores.std()),
        'confidence_stats': {
            'mean_confidence': float(rf_proba.mean()),
            'min_confidence': float(rf_proba.min()),
            'max_confidence': float(rf_proba.max()),
            'high_confidence_ratio': float((rf_proba > 0.9).mean())
        }
    }
    
    models['random_forest'] = rf
    
    logger.info(f"Random Forest - Accuracy: {results['random_forest']['accuracy']:.4f}")
    logger.info(f"Random Forest - F1: {results['random_forest']['f1_score']:.4f}")
    
    # 2. Logistic Regression (regularizado)
    logger.info("Treinando Logistic Regression...")
    start_time = time.time()
    
    lr = LogisticRegression(
        C=0.1,                 # Regularização alta para evitar overfitting
        penalty='l2',
        max_iter=1000,
        random_state=42,
        solver='liblinear'
    )
    lr.fit(X_train, y_train)
    
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    
    lr_time = time.time() - start_time
    
    cv_scores_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='f1')
    
    results['logistic_regression'] = {
        'accuracy': float(accuracy_score(y_test, lr_pred)),
        'f1_score': float(f1_score(y_test, lr_pred)),
        'precision': float(precision_score(y_test, lr_pred)),
        'recall': float(recall_score(y_test, lr_pred)),
        'training_time': lr_time,
        'cv_f1_mean': float(cv_scores_lr.mean()),
        'cv_f1_std': float(cv_scores_lr.std()),
        'confidence_stats': {
            'mean_confidence': float(lr_proba.mean()),
            'min_confidence': float(lr_proba.min()),
            'max_confidence': float(lr_proba.max()),
            'high_confidence_ratio': float((lr_proba > 0.9).mean())
        }
    }
    
    models['logistic_regression'] = lr
    
    logger.info(f"Logistic Regression - Accuracy: {results['logistic_regression']['accuracy']:.4f}")
    logger.info(f"Logistic Regression - F1: {results['logistic_regression']['f1_score']:.4f}")
    
    # 3. Isolation Forest (para detecção de anomalias)
    logger.info("Treinando Isolation Forest...")
    start_time = time.time()
    
    iso_forest = IsolationForest(
        contamination=0.3,     # Expectativa de 30% de ataques
        n_estimators=100,
        max_samples=0.8,       # Usar 80% dos dados por árvore
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    
    iso_pred_raw = iso_forest.predict(X_test)
    iso_pred = np.where(iso_pred_raw == -1, 1, 0)  # -1 = anomalia (ataque), 1 = normal
    iso_scores = iso_forest.decision_function(X_test)
    
    iso_time = time.time() - start_time
    
    results['isolation_forest'] = {
        'accuracy': float(accuracy_score(y_test, iso_pred)),
        'f1_score': float(f1_score(y_test, iso_pred)),
        'precision': float(precision_score(y_test, iso_pred)),
        'recall': float(recall_score(y_test, iso_pred)),
        'training_time': iso_time,
        'anomaly_stats': {
            'mean_score': float(iso_scores.mean()),
            'min_score': float(iso_scores.min()),
            'max_score': float(iso_scores.max()),
            'anomalies_detected': float((iso_pred == 1).mean())
        }
    }
    
    models['isolation_forest'] = iso_forest
    
    logger.info(f"Isolation Forest - Accuracy: {results['isolation_forest']['accuracy']:.4f}")
    logger.info(f"Isolation Forest - F1: {results['isolation_forest']['f1_score']:.4f}")
    
    return models, results

def save_realistic_models(models, results, dataset_name, output_dir):
    """Salvar modelos e resultados"""
    logger.info(f"Salvando modelos para {dataset_name}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Salvar modelos
    for model_name, model in models.items():
        model_file = output_path / f'{model_name}_{dataset_name}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Modelo salvo: {model_file}")
    
    # Salvar resultados
    results_file = output_path / f'results_{dataset_name}.json'
    
    # Adicionar metadados
    final_results = {
        'dataset': dataset_name,
        'version': 'realistic_v2',
        'training_date': pd.Timestamp.now().isoformat(),
        'models': results,
        'summary': {
            'best_accuracy': max([r['accuracy'] for r in results.values()]),
            'best_f1': max([r['f1_score'] for r in results.values()]),
            'total_training_time': sum([r['training_time'] for r in results.values()]),
            'expected_range': '99.0% - 99.8% accuracy'
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Resultados salvos: {results_file}")
    
    return final_results

def main():
    """Função principal"""
    logger.info("Iniciando treinamento com datasets REALISTAS...")
    
    # Diretórios
    data_dir = Path(__file__).parent.parent.parent / 'src' / 'datasets' / 'realistic'
    models_dir = Path(__file__).parent.parent.parent / 'src' / 'models' / 'realistic'
    
    # Verificar se datasets existem
    if not data_dir.exists():
        logger.error(f"Diretório de datasets não encontrado: {data_dir}")
        logger.info("Execute primeiro: ./deployment/scripts/create_realistic_datasets_v2.py")
        return False
    
    all_results = {}
    
    try:
        # Treinar em cada dataset
        for dataset_name in ['cicddos', 'unsw', 'integrated']:
            logger.info(f"\n{'='*50}")
            logger.info(f"TREINANDO DATASET: {dataset_name.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                # Carregar dados
                X_train, X_test, y_train, y_test, metadata = load_realistic_dataset(
                    dataset_name, data_dir
                )
                
                # Treinar modelos
                models, results = train_realistic_models(
                    X_train, X_test, y_train, y_test, dataset_name
                )
                
                # Salvar
                final_results = save_realistic_models(
                    models, results, dataset_name, models_dir
                )
                
                all_results[dataset_name] = final_results
                
                # Log resumo
                best_acc = final_results['summary']['best_accuracy']
                best_f1 = final_results['summary']['best_f1']
                logger.info(f"RESUMO {dataset_name}: Accuracy={best_acc:.4f}, F1={best_f1:.4f}")
                
                if best_acc >= 1.0:
                    logger.warning(f"ALERTA: {dataset_name} ainda com 100% accuracy - revisar dados")
                elif best_acc >= 0.99:
                    logger.info(f"BOM: {dataset_name} com {best_acc:.3f} accuracy (realista)")
                else:
                    logger.warning(f"BAIXO: {dataset_name} com {best_acc:.3f} accuracy")
                
            except Exception as e:
                logger.error(f"Erro ao treinar {dataset_name}: {e}")
                continue
        
        # Resumo geral
        logger.info(f"\n{'='*50}")
        logger.info("RESUMO GERAL")
        logger.info(f"{'='*50}")
        
        for dataset_name, results in all_results.items():
            best_acc = results['summary']['best_accuracy']
            best_f1 = results['summary']['best_f1']
            logger.info(f"{dataset_name:12}: Acc={best_acc:.3f}, F1={best_f1:.3f}")
        
        # Salvar resumo geral
        summary_file = models_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Resumo geral salvo: {summary_file}")
        logger.info("Treinamento concluído com sucesso!")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
