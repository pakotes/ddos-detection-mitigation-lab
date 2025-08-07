#!/usr/bin/env python3
"""
Treinamento SIMPLES com dados REAIS
Usa Random Forest e Logistic Regression nos dados originais
"""

import logging
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def load_real_dataset(dataset_name, data_dir):
    """Carregar dataset real"""
    logger.info(f"Carregando dataset real: {dataset_name}")
    
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
    logger.info(f"  Taxa de ataques: {metadata.get('attack_ratio', 'N/A'):.1%}")
    
    return X_train, X_test, y_train, y_test, metadata

def train_simple_models(X_train, X_test, y_train, y_test, dataset_name):
    """Treinar modelos simples nos dados reais"""
    logger.info(f"Treinando modelos para {dataset_name}...")
    
    models = {}
    results = {}
    
    # 1. Random Forest (configuração conservadora)
    logger.info("Treinando Random Forest...")
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,        # Deixar crescer naturalmente
        min_samples_split=2,   # Configuração padrão
        min_samples_leaf=1,    # Configuração padrão  
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    rf_pred = rf.predict(X_test)
    rf_time = time.time() - start_time
    
    results['random_forest'] = {
        'accuracy': float(accuracy_score(y_test, rf_pred)),
        'f1_score': float(f1_score(y_test, rf_pred)),
        'training_time': rf_time
    }
    
    models['random_forest'] = rf
    
    logger.info(f"Random Forest - Accuracy: {results['random_forest']['accuracy']:.4f}")
    logger.info(f"Random Forest - F1: {results['random_forest']['f1_score']:.4f}")
    
    # 2. Logistic Regression (configuração padrão)
    logger.info("Treinando Logistic Regression...")
    start_time = time.time()
    
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='liblinear'
    )
    lr.fit(X_train, y_train)
    
    lr_pred = lr.predict(X_test)
    lr_time = time.time() - start_time
    
    results['logistic_regression'] = {
        'accuracy': float(accuracy_score(y_test, lr_pred)),
        'f1_score': float(f1_score(y_test, lr_pred)),
        'training_time': lr_time
    }
    
    models['logistic_regression'] = lr
    
    logger.info(f"Logistic Regression - Accuracy: {results['logistic_regression']['accuracy']:.4f}")
    logger.info(f"Logistic Regression - F1: {results['logistic_regression']['f1_score']:.4f}")
    
    # 3. Relatório detalhado para o melhor modelo
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = models[best_model_name]
    best_pred = best_model.predict(X_test)
    
    logger.info(f"\nRelatório detalhado do melhor modelo ({best_model_name}):")
    report = classification_report(y_test, best_pred, target_names=['Normal', 'Ataque'])
    logger.info(f"\n{report}")
    
    return models, results

def save_real_models(models, results, dataset_name, output_dir):
    """Salvar modelos e resultados"""
    logger.info(f"Salvando modelos para {dataset_name}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Salvar modelos
    for model_name, model in models.items():
        model_file = output_path / f'{model_name}_{dataset_name}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
    
    # Salvar resultados
    results_file = output_path / f'results_{dataset_name}.json'
    
    final_results = {
        'dataset': dataset_name,
        'source': 'REAL_DATA',
        'training_date': pd.Timestamp.now().isoformat(),
        'models': results,
        'summary': {
            'best_accuracy': max([r['accuracy'] for r in results.values()]),
            'best_f1': max([r['f1_score'] for r in results.values()]),
            'total_training_time': sum([r['training_time'] for r in results.values()])
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Modelos salvos em: {output_path}")
    
    return final_results

def main():
    """Função principal"""
    logger.info("Iniciando treinamento SIMPLES com dados REAIS...")
    
    # Diretórios
    data_dir = Path(__file__).parent.parent.parent / 'src' / 'datasets' / 'real'
    models_dir = Path(__file__).parent.parent.parent / 'src' / 'models' / 'real'
    
    # Verificar se datasets existem
    if not data_dir.exists():
        logger.error(f"Diretório de datasets não encontrado: {data_dir}")
        logger.info("Execute primeiro: ddos-process-real")
        return False
    
    all_results = {}
    
    try:
        # Treinar em cada dataset disponível
        for dataset_name in ['cicddos', 'unsw']:
            logger.info(f"\n{'='*50}")
            logger.info(f"TREINANDO DATASET REAL: {dataset_name.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                # Carregar dados
                X_train, X_test, y_train, y_test, metadata = load_real_dataset(
                    dataset_name, data_dir
                )
                
                # Treinar modelos
                models, results = train_simple_models(
                    X_train, X_test, y_train, y_test, dataset_name
                )
                
                # Salvar
                final_results = save_real_models(
                    models, results, dataset_name, models_dir
                )
                
                all_results[dataset_name] = final_results
                
                # Log resumo
                best_acc = final_results['summary']['best_accuracy']
                best_f1 = final_results['summary']['best_f1']
                logger.info(f"RESUMO {dataset_name}: Accuracy={best_acc:.4f}, F1={best_f1:.4f}")
                
            except Exception as e:
                logger.error(f"Erro ao treinar {dataset_name}: {e}")
                continue
        
        # Resumo geral
        logger.info(f"\n{'='*50}")
        logger.info("RESUMO GERAL - DADOS REAIS")
        logger.info(f"{'='*50}")
        
        for dataset_name, results in all_results.items():
            best_acc = results['summary']['best_accuracy']
            best_f1 = results['summary']['best_f1']
            logger.info(f"{dataset_name:12}: Acc={best_acc:.3f}, F1={best_f1:.3f}")
        
        # Salvar resumo geral
        if all_results:
            summary_file = models_dir / 'training_summary_real.json'
            with open(summary_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            logger.info(f"Resumo salvo: {summary_file}")
        
        logger.info("✅ Treinamento com dados REAIS concluído!")
        logger.info("📊 Resultados devem ser realistas (90-99% accuracy)")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
