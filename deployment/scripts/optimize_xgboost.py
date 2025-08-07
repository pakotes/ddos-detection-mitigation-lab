#!/usr/bin/env python3
"""
Otimização de Hiperparâmetros para XGBoost
Melhoria de performance e redução de overfitting
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import pickle
import time
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import xgboost as xgb
import optuna
from scipy.stats import uniform, randint

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_for_tuning():
    """Carrega dados para tuning"""
    logger.info("🚀 Carregando dados para tuning...")
    
    # Tentar carregar dados avançados primeiro
    advanced_path = Path("./datasets/integrated")
    
    files_to_try = [
        ("X_integrated_advanced.npy", "y_integrated_advanced.npy", "avançados"),
        ("X_integrated_real.npy", "y_integrated_real.npy", "reais")
    ]
    
    for x_file, y_file, desc in files_to_try:
        X_path = advanced_path / x_file
        y_path = advanced_path / y_file
        
        if X_path.exists() and y_path.exists():
            X = np.load(X_path)
            y = np.load(y_path)
            logger.info(f"Dados {desc} carregados: {X.shape}")
            return X, y, desc
    
    raise FileNotFoundError("Nenhum dataset encontrado para tuning!")

def optimize_xgboost_bayesian(X, y, n_trials=100):
    """Otimização Bayesiana do XGBoost usando Optuna"""
    logger.info(f"🔍 Iniciando otimização Bayesiana ({n_trials} trials)...")
    
    def objective(trial):
        # Espaço de busca otimizado
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'random_state': 42,
            
            # Parâmetros de árvore
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            
            # Parâmetros de regularização
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            
            # Parâmetros de learning
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            
            # Parâmetros avançados
            'gamma': trial.suggest_float('gamma', 0, 5),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }
        
        # Validação cruzada
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)], 
                     early_stopping_rounds=50,
                     verbose=False)
            
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    # Executar otimização
    study = optuna.create_study(direction='maximize', 
                               sampler=optuna.samplers.TPESampler(seed=42))
    
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    end_time = time.time()
    
    logger.info(f"⏱️ Otimização concluída em {end_time - start_time:.2f}s")
    logger.info(f"🏆 Melhor F1-Score: {study.best_value:.6f}")
    logger.info(f"🎯 Melhores parâmetros: {study.best_params}")
    
    return study.best_params, study.best_value

def optimize_xgboost_grid(X, y):
    """Otimização por Grid Search (mais conservadora)"""
    logger.info("🔍 Executando Grid Search otimizado...")
    
    # Grid mais focado em performance alta
    param_grid = {
        'max_depth': [4, 6, 8],
        'min_child_weight': [1, 3, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [300, 500, 700],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 2, 5],
        'scale_pos_weight': [1, 3, 5]
    }
    
    # Usar RandomizedSearchCV para eficiência
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=50,  # 50 combinações aleatórias
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    start_time = time.time()
    random_search.fit(X, y)
    end_time = time.time()
    
    logger.info(f"⏱️ Random Search concluído em {end_time - start_time:.2f}s")
    logger.info(f"🏆 Melhor F1-Score: {random_search.best_score_:.6f}")
    logger.info(f"🎯 Melhores parâmetros: {random_search.best_params_}")
    
    return random_search.best_params_, random_search.best_score_

def train_optimized_model(X, y, best_params):
    """Treina modelo com parâmetros otimizados"""
    logger.info("🏋️ Treinando modelo otimizado...")
    
    # Adicionar parâmetros fixos
    final_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
        **best_params
    }
    
    # Dividir dados para treino/validação
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Treinar modelo
    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             early_stopping_rounds=100,
             verbose=True)
    
    # Avaliar
    y_pred = model.predict(X_test)
    
    metrics = {
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    logger.info(f"🎯 F1-Score final: {metrics['f1_score']:.6f}")
    logger.info(f"🎯 Precision: {metrics['precision']:.6f}")
    logger.info(f"🎯 Recall: {metrics['recall']:.6f}")
    
    return model, metrics

def save_optimized_xgboost(model, params, metrics, data_type, method, output_dir="./models/hybrid_advanced"):
    """Salva modelo XGBoost otimizado"""
    logger.info("💾 Salvando modelo XGBoost otimizado...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Nome do arquivo baseado no método
    model_name = f"xgboost_optimized_{method}_{data_type}.pkl"
    results_name = f"xgboost_optimization_{method}_{data_type}.json"
    
    # Salvar modelo
    model_path = output_path / model_name
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Salvar resultados
    optimization_results = {
        'method': method,
        'data_type': data_type,
        'best_params': params,
        'metrics': metrics,
        'optimization_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = output_path / results_name
    with open(results_path, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    logger.info(f"✅ Modelo salvo em: {model_path}")
    logger.info(f"✅ Resultados salvos em: {results_path}")

def main():
    """Função principal"""
    try:
        logger.info("🚀 Iniciando otimização do XGBoost...")
        
        # 1. Carregar dados
        X, y, data_type = load_data_for_tuning()
        
        # 2. Escolher método de otimização
        methods = {
            'bayesian': optimize_xgboost_bayesian,
            'grid': optimize_xgboost_grid
        }
        
        results = {}
        
        for method_name, optimize_func in methods.items():
            logger.info(f"🔍 Executando otimização: {method_name}")
            
            try:
                if method_name == 'bayesian':
                    best_params, best_score = optimize_func(X, y, n_trials=50)
                else:
                    best_params, best_score = optimize_func(X, y)
                
                # Treinar modelo otimizado
                model, metrics = train_optimized_model(X, y, best_params)
                
                # Salvar resultados
                save_optimized_xgboost(model, best_params, metrics, data_type, method_name)
                
                results[method_name] = {
                    'params': best_params,
                    'metrics': metrics,
                    'cv_score': best_score
                }
                
            except Exception as e:
                logger.error(f"❌ Erro na otimização {method_name}: {e}")
                continue
        
        # Comparar resultados
        if results:
            logger.info("📊 COMPARAÇÃO DE RESULTADOS:")
            for method, result in results.items():
                logger.info(f"{method}: F1 = {result['metrics']['f1_score']:.6f}")
        
        logger.info("✅ Otimização do XGBoost concluída!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na otimização: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
