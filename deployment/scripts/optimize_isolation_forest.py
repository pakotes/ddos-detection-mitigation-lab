#!/usr/bin/env python3
"""
Otimização do Isolation Forest para melhor detecção de anomalias
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import json
import pickle
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_integrated_data():
    """Carrega dados integrados do UNSW-NB15"""
    logger.info("🚀 Carregando dados integrados UNSW-NB15...")
    
    base_path = Path("./datasets/integrated")
    
    # Tentar carregar dados avançados primeiro
    files_to_try = [
        ("X_integrated_advanced.npy", "y_integrated_advanced.npy", "avançados"),
        ("X_integrated_real.npy", "y_integrated_real.npy", "básicos")
    ]
    
    for x_file, y_file, desc in files_to_try:
        X_path = base_path / x_file
        y_path = base_path / y_file
        
        if X_path.exists() and y_path.exists():
            X = np.load(X_path)
            y = np.load(y_path)
            logger.info(f"Dados {desc} carregados: {X.shape} samples, {X.shape[1]} features")
            logger.info(f"Distribuição: Normal={np.sum(y == 0)}, Attack={np.sum(y == 1)}")
            return X, y
    
    raise FileNotFoundError("Dados integrados não encontrados! Execute setup primeiro.")

def optimize_isolation_forest(X, y):
    """Otimiza hiperparâmetros do Isolation Forest"""
    logger.info("🔧 Otimizando hiperparâmetros do Isolation Forest...")
    
    # Grid de hiperparâmetros mais focado
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_samples': [0.7, 0.8, 0.9],
        'contamination': [0.05, 0.1, 0.15],  # Baseado na proporção real de ataques
        'max_features': [0.8, 1.0],
        'bootstrap': [True, False],
        'random_state': [42]
    }
    
    # Função de scoring personalizada para Isolation Forest
    def if_scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        # Converter -1/1 para 0/1
        y_pred_binary = np.where(y_pred == -1, 1, 0)  # -1 = anomaly = attack
        return f1_score(y, y_pred_binary)
    
    # Grid Search com validação cruzada mais simples
    logger.info("🔍 Executando Grid Search otimizado...")
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # Reduzido para 2 folds
    
    grid_search = GridSearchCV(
        IsolationForest(),
        param_grid,
        scoring=if_scorer,
        cv=cv,
        n_jobs=2,  # Reduzido para evitar overhead
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X, y)
    end_time = time.time()
    
    logger.info(f"⏱️ Grid Search concluído em {end_time - start_time:.2f}s")
    logger.info(f"🏆 Melhor score: {grid_search.best_score_:.4f}")
    logger.info(f"🎯 Melhores parâmetros: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def evaluate_model(model, X, y):
    """Avalia o modelo otimizado"""
    logger.info("📊 Avaliando modelo otimizado...")
    
    # Predição
    y_pred = model.predict(X)
    y_pred_binary = np.where(y_pred == -1, 1, 0)  # -1 = anomaly = attack
    
    # Métricas
    f1 = f1_score(y, y_pred_binary)
    precision = precision_score(y, y_pred_binary)
    recall = recall_score(y, y_pred_binary)
    
    logger.info(f"🎯 F1-Score: {f1:.4f}")
    logger.info(f"🎯 Precision: {precision:.4f}")
    logger.info(f"🎯 Recall: {recall:.4f}")
    
    # Relatório detalhado
    logger.info("📋 Relatório de classificação:")
    report = classification_report(y, y_pred_binary)
    logger.info(f"\n{report}")
    
    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'classification_report': report
    }

def save_optimized_model(model, params, metrics, output_dir="./models/hybrid_advanced"):
    """Salva o modelo otimizado"""
    logger.info("💾 Salvando modelo otimizado...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Salvar modelo
    model_path = output_path / "isolation_forest_optimized.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Salvar parâmetros e métricas
    optimization_results = {
        'best_params': params,
        'metrics': metrics,
        'optimization_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = output_path / "isolation_forest_optimization.json"
    with open(results_path, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    logger.info(f"✅ Modelo salvo em: {model_path}")
    logger.info(f"✅ Resultados salvos em: {results_path}")

def main():
    """Função principal"""
    try:
        logger.info("🚀 Iniciando otimização do Isolation Forest...")
        
        # 1. Carregar dados
        X, y = load_integrated_data()
        
        # 2. Otimizar modelo
        best_model, best_params, best_score = optimize_isolation_forest(X, y)
        
        # 3. Avaliar modelo otimizado
        metrics = evaluate_model(best_model, X, y)
        
        # 4. Salvar resultados
        save_optimized_model(best_model, best_params, metrics)
        
        logger.info("✅ Otimização concluída com sucesso!")
        logger.info(f"📈 Melhoria: F1-Score otimizado = {metrics['f1_score']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na otimização: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
