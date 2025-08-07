#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Treinamento e Avaliacao para Modelos de Deteccao DDoS

Este modulo implementa:
1. Carregamento do dataset CIC-DDoS2019
2. Treinamento dos modelos hibridos
3. Avaliacao com metricas especificas (Recall, Precisao, F1-Score, Falsos Positivos)
4. Validacao cruzada
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.datasets import make_classification
import xgboost as xgb
import pickle
import time
from datetime import datetime
import json
import structlog
from typing import Dict, Tuple

logger = structlog.get_logger()


class SyntheticDataGenerator:
    """Gerador de dados sinteticos para demonstracao (fallback)"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def generate_ddos_dataset(self, n_samples: int = 10000, n_features: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Gera dataset sintetico simulando trafego DDoS apenas como fallback"""
        logger.warning("Usando dados sinteticos - ideal e usar CIC-DDoS2019 real")
        
        # Gerar dados base
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=3,
            n_clusters_per_class=2,
            class_sep=0.8,
            random_state=self.random_state
        )
        
        # Simular caracteristicas de trafego de rede
        feature_names = [
            'packet_size', 'flow_duration', 'packet_rate', 'byte_rate',
            'protocol_type', 'src_port', 'dst_port', 'tcp_flags',
            'packet_count', 'syn_count', 'ack_count', 'fin_count',
            'inter_arrival_time', 'flow_bytes_per_second', 'flow_packets_per_second',
            'payload_entropy', 'header_length', 'window_size', 'ttl', 'fragmentation'
        ]
        
        # Converter para DataFrame
        df = pd.DataFrame(X, columns=feature_names[:n_features])
        
        # Adicionar características realísticas
        for i in range(len(df)):
            if y[i] == 1:  # Tráfego malicioso
                df.loc[i, 'packet_rate'] += np.random.normal(50, 10)
                df.loc[i, 'syn_count'] += np.random.normal(100, 20)
                df.loc[i, 'flow_duration'] *= 0.1
            else:  # Tráfego benigno
                df.loc[i, 'packet_rate'] += np.random.normal(5, 2)
                df.loc[i, 'flow_duration'] += np.random.normal(10, 5)
        
        # Garantir valores positivos
        positive_features = ['packet_size', 'flow_duration', 'packet_rate', 'byte_rate', 'packet_count']
        for feature in positive_features:
            if feature in df.columns:
                df[feature] = np.abs(df[feature])
        
        logger.info("Dataset sintético gerado", 
                   benign_samples=(y == 0).sum(),
                   malicious_samples=(y == 1).sum())
        
        return df.values, y


class CICDDoS2019Loader:
    """Carregador do dataset CICDDoS2019"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.label_mapping = {
            # Trafego benigno
            'Benign': 0,
            'BENIGN': 0,
            
            # Ataques DDoS (todos como maliciosos = 1)
            'DrDoS_DNS': 1,
            'DrDoS_LDAP': 1,
            'DrDoS_MSSQL': 1,
            'DrDoS_NetBIOS': 1,
            'DrDoS_NTP': 1,
            'DrDoS_SNMP': 1,
            'DrDoS_UDP': 1,
            'LDAP': 1,
            'MSSQL': 1,
            'NetBIOS': 1,
            'Portmap': 1,
            'Syn': 1,
            'TFTP': 1,
            'UDP-lag': 1,
            'UDP': 1,
            'UDPLag': 1,
            'WebDDoS': 1
        }
    
    def load_dataset(self) -> tuple:
        """Carrega e preprocessa o dataset CICDDoS2019"""
        logger.info("Carregando dataset CICDDoS2019", path=self.dataset_path)
        
        # Carregar todos os arquivos CSV do dataset
        csv_files = []
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado in {self.dataset_path}")
        
        # Combinar todos os arquivos
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dataframes.append(df)
                logger.info(f"Carregado {csv_file}", shape=df.shape)
            except Exception as e:
                logger.warning(f"Erro ao carregar {csv_file}", error=str(e))
        
        # Combinar dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info("Dataset combinado", total_shape=combined_df.shape)
        
        # Preprocessar dados
        X, y = self._preprocess_data(combined_df)
        
        return X, y
    
    def _preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocessa os dados do CICDDoS2019"""
        logger.info("Preprocessando dados...")
        
        # Remover colunas não numéricas e irrelevantes
        columns_to_drop = ['Timestamp', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Separar features e labels
        if 'Label' in df.columns:
            y = df['Label'].map(self.label_mapping).fillna(1)  # Desconhecidos como maliciosos
            X = df.drop('Label', axis=1)
        else:
            raise ValueError("Coluna 'Label' não encontrada no dataset")
        
        # Remover linhas com valores infinitos ou NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Converter tudo para numérico
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        logger.info("Preprocessamento concluído", 
                   features_shape=X.shape, 
                   labels_shape=y.shape,
                   malicious_ratio=y.mean())
        
        return X.values, y.values


class ModelTrainer:
    """Treinador de modelos híbridos"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.metrics = {}
    
    def train_supervised_models(self, X_train, y_train, X_test, y_test):
        """Treina modelos supervisionados (XGBoost e Random Forest)"""
        logger.info("Treinando modelos supervisionados...")
        
        # Normalizar dados
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar XGBoost
        logger.info("Treinando XGBoost...")
        start_time = time.time()
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train_scaled, y_train)
        xgb_train_time = time.time() - start_time
        
        # Avaliar XGBoost
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
        xgb_metrics = self._calculate_metrics(y_test, xgb_pred, xgb_pred_proba, "XGBoost", xgb_train_time)
        
        self.models['xgboost'] = xgb_model
        self.metrics['xgboost'] = xgb_metrics
        
        # Treinar Random Forest
        logger.info("Treinando Random Forest...")
        start_time = time.time()
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        rf_train_time = time.time() - start_time
        
        # Avaliar Random Forest
        rf_pred = rf_model.predict(X_test_scaled)
        rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        rf_metrics = self._calculate_metrics(y_test, rf_pred, rf_pred_proba, "Random Forest", rf_train_time)
        
        self.models['random_forest'] = rf_model
        self.metrics['random_forest'] = rf_metrics
        
        return xgb_metrics, rf_metrics
    
    def train_unsupervised_model(self, X_train, X_test, y_test):
        """Treina modelo não-supervisionado (Isolation Forest)"""
        logger.info("Treinando Isolation Forest...")
        start_time = time.time()
        
        # Usar apenas dados benignos para treinar (unsupervised)
        X_benign = X_train[y_train == 0] if 'y_train' in locals() else X_train
        X_train_scaled = self.scaler.transform(X_benign)
        X_test_scaled = self.scaler.transform(X_test)
        
        if_model = IsolationForest(
            contamination=0.1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        if_model.fit(X_train_scaled)
        if_train_time = time.time() - start_time
        
        # Avaliar Isolation Forest
        if_pred = if_model.predict(X_test_scaled)
        if_pred = np.where(if_pred == -1, 1, 0)  # -1 (anomalia) -> 1 (malicioso)
        if_scores = if_model.decision_function(X_test_scaled)
        if_scores_normalized = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
        
        if_metrics = self._calculate_metrics(y_test, if_pred, if_scores_normalized, "Isolation Forest", if_train_time)
        
        self.models['isolation_forest'] = if_model
        self.metrics['isolation_forest'] = if_metrics
        
        return if_metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, model_name, train_time):
        """Calcula métricas detalhadas do modelo"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Matriz de confusão
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Taxa de falsos positivos
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc_roc = 0.0
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'false_positive_rate': fpr,
            'auc_roc': auc_roc,
            'training_time_seconds': train_time
        }
        
        logger.info(f"Métricas {model_name}", **metrics)
        
        return metrics
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Validação cruzada dos modelos"""
        logger.info("Executando validação cruzada...")
        
        X_scaled = self.scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            if model_name == 'isolation_forest':
                continue  # Skip unsupervised model
            
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
            cv_results[model_name] = {
                'mean_f1': scores.mean(),
                'std_f1': scores.std(),
                'scores': scores.tolist()
            }
            
            logger.info(f"Validação cruzada {model_name}", 
                       mean_f1=scores.mean(), 
                       std_f1=scores.std())
        
        return cv_results
    
    def save_models(self, model_path: str):
        """Salva todos os modelos treinados"""
        os.makedirs(model_path, exist_ok=True)
        
        # Salvar modelos
        for model_name, model in self.models.items():
            with open(f"{model_path}/{model_name}.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        # Salvar scaler
        with open(f"{model_path}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Salvar métricas
        with open(f"{model_path}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        logger.info("Modelos salvos", path=model_path)
    
    def generate_evaluation_report(self) -> Dict:
        """Gera relatório completo de avaliação"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': len(self.models),
            'metrics_summary': {},
            'detailed_metrics': self.metrics
        }
        
        # Resumo comparativo
        for model_name, metrics in self.metrics.items():
            report['metrics_summary'][model_name] = {
                'f1_score': metrics.get('f1_score', 0),
                'recall': metrics.get('recall', 0),
                'precision': metrics.get('precision', 0),
                'false_positive_rate': metrics.get('false_positive_rate', 0),
                'training_time': metrics.get('training_time_seconds', 0)
            }
        
        return report


def main():
    """Função principal para treinamento e avaliação"""
    # Configuração
    dataset_path = "/app/datasets/cicddos2019"
    model_path = "/app/models"
    
    logger.info("Iniciando treinamento de modelos ML DDoS")
    
    # Tentar carregar dataset CIC-DDoS2019 real
    try:
        if os.path.exists(dataset_path) and os.listdir(dataset_path):
            logger.info("Dataset CIC-DDoS2019 encontrado, carregando...")
            loader = CICDDoS2019Loader(dataset_path)
            X, y = loader.load_dataset()
            logger.info("Dataset CIC-DDoS2019 carregado com sucesso", 
                       samples=len(X), features=X.shape[1])
        else:
            raise FileNotFoundError("Dataset CIC-DDoS2019 não encontrado")
            
    except Exception as e:
        logger.warning(f"Não foi possível carregar CIC-DDoS2019: {e}")
        logger.info("INSTRUÇÕES PARA DOWNLOAD DO DATASET CIC-DDoS2019:")
        logger.info("1. Baixe de: https://www.unb.ca/cic/datasets/ddos-2019.html")
        logger.info("2. Extraia os arquivos CSV para: /app/datasets/cicddos2019/")
        logger.info("3. Execute novamente o treinamento")
        logger.info("")
        logger.info("Por enquanto, usando dados sintéticos para demonstração...")
        
        # Fallback para dados sintéticos
        synthetic_generator = SyntheticDataGenerator()
        X, y = synthetic_generator.generate_ddos_dataset(n_samples=5000, n_features=20)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info("Dados divididos", 
               train_samples=len(X_train),
               test_samples=len(X_test),
               train_malicious_ratio=y_train.mean(),
               test_malicious_ratio=y_test.mean())
    
    # Treinar modelos
    trainer = ModelTrainer()
    
    # Modelos supervisionados
    logger.info("=== TREINANDO MODELOS SUPERVISIONADOS ===")
    xgb_metrics, rf_metrics = trainer.train_supervised_models(X_train, y_train, X_test, y_test)
    
    # Modelo não-supervisionado  
    logger.info("=== TREINANDO MODELO NÃO-SUPERVISIONADO ===")
    if_metrics = trainer.train_unsupervised_model(X_train, X_test, y_test)
    
    # Validação cruzada
    logger.info("=== EXECUTANDO VALIDAÇÃO CRUZADA ===")
    cv_results = trainer.cross_validate_models(X, y)
    
    # Salvar modelos e métricas
    trainer.save_models(model_path)
    
    # Gerar relatório
    report = trainer.generate_evaluation_report()
    report['cross_validation'] = cv_results
    report['dataset_info'] = {
        'total_samples': len(X),
        'features': X.shape[1],
        'malicious_ratio': float(y.mean()),
        'data_source': 'CIC-DDoS2019' if os.path.exists(dataset_path) else 'synthetic'
    }
    
    # Salvar relatório
    report_file = f"{model_path}/evaluation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Mostrar resumo dos resultados
    logger.info("=== RESUMO DOS RESULTADOS ===")
    for model_name, metrics in trainer.metrics.items():
        logger.info(f"{model_name.upper()}: F1={metrics['f1_score']:.3f}, "
                   f"Precision={metrics['precision']:.3f}, "
                   f"Recall={metrics['recall']:.3f}, "
                   f"FPR={metrics['false_positive_rate']:.3f}")
    
    logger.info("Treinamento concluído", 
               models_trained=len(trainer.models),
               report_path=report_file)
    
    return report


if __name__ == "__main__":
    main()
