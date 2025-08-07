#!/usr/bin/env python3
"""
Otimização Simplificada dos Modelos
Versão rápida e eficiente para melhorar os modelos existentes
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import pickle
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import xgboost as xgb

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_best_data():
    """Carrega os melhores dados disponíveis"""
    logger.info("🚀 Carregando melhores dados disponíveis...")
    
    base_path = Path("./datasets/integrated")
    
    # Tentar dados avançados primeiro
    files_to_try = [
        ("X_integrated_advanced.npy", "y_integrated_advanced.npy", "avançados (89 features)"),
        ("X_integrated_real.npy", "y_integrated_real.npy", "básicos (50 features)")
    ]
    
    for x_file, y_file, desc in files_to_try:
        X_path = base_path / x_file
        y_path = base_path / y_file
        
        if X_path.exists() and y_path.exists():
            X = np.load(X_path)
            y = np.load(y_path)
            logger.info(f"✅ Dados {desc} carregados: {X.shape}")
            logger.info(f"📊 Distribuição: Normal={np.sum(y == 0)}, Attack={np.sum(y == 1)}")
            return X, y, desc
    
    raise FileNotFoundError("Nenhum dataset encontrado!")

def optimize_isolation_forest_simple(X, y):
    """Otimização simplificada do Isolation Forest"""
    logger.info("🎯 Otimizando Isolation Forest (método simplificado)...")
    
    # Testar diferentes configurações
    configs = [
        {'contamination': 0.05, 'n_estimators': 200, 'max_features': 0.8},
        {'contamination': 0.1, 'n_estimators': 300, 'max_features': 1.0},
        {'contamination': 0.15, 'n_estimators': 200, 'max_features': 0.9},
        {'contamination': 0.08, 'n_estimators': 250, 'max_features': 0.85}
    ]
    
    best_score = 0
    best_config = None
    best_model = None
    
    # Dividir dados para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    for i, config in enumerate(configs):
        logger.info(f"🔍 Testando configuração {i+1}/4: {config}")
        
        model = IsolationForest(random_state=42, **config)
        model.fit(X_train)
        
        y_pred = model.predict(X_test)
        y_pred_binary = np.where(y_pred == -1, 1, 0)  # -1 = anomaly = attack
        
        score = f1_score(y_test, y_pred_binary)
        logger.info(f"   F1-Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_config = config
            best_model = model
    
    logger.info(f"🏆 Melhor configuração: {best_config}")
    logger.info(f"🎯 Melhor F1-Score: {best_score:.4f}")
    
    return best_model, best_config, best_score

def optimize_xgboost_simple(X, y):
    """Otimização simplificada do XGBoost"""
    logger.info("🎯 Otimizando XGBoost (método simplificado)...")
    
    # Configurações testadas
    configs = [
        {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 300, 'subsample': 0.8},
        {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 500, 'subsample': 0.9},
        {'max_depth': 4, 'learning_rate': 0.2, 'n_estimators': 200, 'subsample': 0.85},
        {'max_depth': 7, 'learning_rate': 0.08, 'n_estimators': 400, 'subsample': 0.9}
    ]
    
    best_score = 0
    best_config = None
    best_model = None
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    for i, config in enumerate(configs):
        logger.info(f"🔍 Testando XGBoost {i+1}/4: {config}")
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            **config
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        score = f1_score(y_test, y_pred)
        logger.info(f"   F1-Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_config = config
            best_model = model
    
    logger.info(f"🏆 Melhor XGBoost: {best_config}")
    logger.info(f"🎯 Melhor F1-Score: {best_score:.4f}")
    
    return best_model, best_config, best_score

def evaluate_model_complete(model, X, y, model_name):
    """Avaliação completa do modelo"""
    logger.info(f"📊 Avaliação completa: {model_name}")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Treinar se necessário
    if hasattr(model, 'fit') and model_name != "Isolation Forest":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        # Para Isolation Forest
        if model_name == "Isolation Forest":
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == -1, 1, 0)
        else:
            y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = {
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    logger.info(f"🎯 {model_name} - F1: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
    return metrics

def save_optimized_results(models_results, data_desc):
    """Salva resultados da otimização"""
    logger.info("💾 Salvando resultados otimizados...")
    
    output_dir = Path("./models/hybrid_advanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar modelos
    for model_name, (model, config, metrics) in models_results.items():
        model_file = output_dir / f"{model_name.lower().replace(' ', '_')}_optimized_simple.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✅ {model_name} salvo: {model_file}")
    
    # Salvar resultados consolidados
    summary = {
        'data_type': data_desc,
        'optimization_method': 'simplified_grid_search',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': {}
    }
    
    for model_name, (model, config, metrics) in models_results.items():
        summary['results'][model_name] = {
            'config': config,
            'metrics': metrics
        }
    
    summary_file = output_dir / "optimization_summary_simple.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Resumo salvo: {summary_file}")

def main():
    """Função principal"""
    try:
        logger.info("🚀 Iniciando otimização simplificada dos modelos...")
        
        # 1. Carregar dados
        X, y, data_desc = load_best_data()
        
        # 2. Otimizar modelos
        models_results = {}
        
        # Isolation Forest
        try:
            if_model, if_config, if_score = optimize_isolation_forest_simple(X, y)
            if_metrics = evaluate_model_complete(if_model, X, y, "Isolation Forest")
            models_results["Isolation Forest"] = (if_model, if_config, if_metrics)
        except Exception as e:
            logger.error(f"❌ Erro no Isolation Forest: {e}")
        
        # XGBoost
        try:
            xgb_model, xgb_config, xgb_score = optimize_xgboost_simple(X, y)
            xgb_metrics = evaluate_model_complete(xgb_model, X, y, "XGBoost")
            models_results["XGBoost"] = (xgb_model, xgb_config, xgb_metrics)
        except Exception as e:
            logger.error(f"❌ Erro no XGBoost: {e}")
        
        # 3. Salvar resultados
        if models_results:
            save_optimized_results(models_results, data_desc)
            
            # Mostrar resumo
            logger.info("📈 RESUMO DOS RESULTADOS:")
            for model_name, (model, config, metrics) in models_results.items():
                logger.info(f"  {model_name}: F1 = {metrics['f1_score']:.4f}")
        
        logger.info("✅ Otimização simplificada concluída!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na otimização: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
