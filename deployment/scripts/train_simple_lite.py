#!/usr/bin/env python3
"""
Treinador de Modelos Simples - Para Sistemas com Pouca Memória

Treina modelos básicos com pegada mínima de memória para sistemas com RAM limitada.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_lite_model():
    """Treina modelo lite com uso mínimo de memória"""
    logger.info("Iniciando treinamento de modelo lite (modo baixa memória)...")
    
    start_time = time.time()
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
        
        # Gerar dados sintéticos se dados reais muito grandes
        logger.info("Gerando dados sintéticos para eficiência de memória...")
        
        # Criar dataset sintético menor (10k amostras)
        np.random.seed(42)
        n_samples = 10000
        n_features = 20
        
        # Gerar features simulando tráfego de rede
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Gerar labels realistas (20% ataques, 80% benigno)
        y = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        
        # Adicionar correlação para tornar mais realista
        attack_mask = y == 1
        X[attack_mask, :5] += 2.0  # Tráfego de ataque tem valores maiores nas primeiras 5 features
        
        logger.info(f"Dataset sintético criado: {X.shape}")
        logger.info(f"Distribuição de classes: {np.bincount(y)}")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Treinar Random Forest simples (eficiente em memória)
        logger.info("Treinando Random Forest (versão lite)...")
        rf_model = RandomForestClassifier(
            n_estimators=20,    # Muito pequeno para eficiência de memória
            max_depth=5,        # Árvores rasas
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1           # Single thread
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Avaliar
        rf_pred = rf_model.predict(X_test_scaled)
        rf_f1 = f1_score(y_test, rf_pred)
        rf_precision = precision_score(y_test, rf_pred)
        rf_recall = recall_score(y_test, rf_pred)
        
        logger.info(f"Random Forest Lite - F1: {rf_f1:.4f}, Precisão: {rf_precision:.4f}, Recall: {rf_recall:.4f}")
        
        # Salvar modelo
        models_dir = Path("src/models/simple")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': rf_model,
            'scaler': scaler,
            'feature_names': [f'feature_{i}' for i in range(n_features)]
        }
        
        model_path = models_dir / "rf_model_lite.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Salvar métricas de desempenho
        performance = {
            'model_type': 'random_forest_lite',
            'f1_score': float(rf_f1),
            'precision': float(rf_precision),
            'recall': float(rf_recall),
            'training_time': time.time() - start_time,
            'dataset_size': X.shape,
            'synthetic_data': True
        }
        
        perf_path = models_dir / "performance_lite.json"
        with open(perf_path, 'w') as f:
            json.dump(performance, f, indent=2)
        
        logger.info(f"Modelo lite salvo em: {model_path}")
        logger.info(f"Desempenho salvo em: {perf_path}")
        logger.info(f"Treinamento concluído em {time.time() - start_time:.2f} segundos")
        
        return True
        
    except Exception as e:
        logger.error(f"Treinamento falhou: {e}")
        return False

if __name__ == "__main__":
    success = train_lite_model()
    if success:
        logger.info("✅ Treinamento de modelo lite concluído com sucesso")
    else:
        logger.error("❌ Treinamento de modelo lite falhou")
        sys.exit(1)
