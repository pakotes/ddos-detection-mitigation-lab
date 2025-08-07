#!/usr/bin/env python3
"""
Hybrid Model Trainer - Versão Avançada

Usa os dados com feature engineering avançado do UNSW-NB15
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import time
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_advanced_unsw_data():
    """Carrega dados UNSW-NB15 com feature engineering avançado"""
    
    # Tentar diferentes localizações para dados integrados
    possible_integrated_paths = [
        Path("./src/datasets/integrated"),
        Path("./datasets/integrated"),
        Path("../src/datasets/integrated"),
        Path("../../src/datasets/integrated")
    ]
    
    integrated_path = None
    for path in possible_integrated_paths:
        if path.exists():
            integrated_path = path
            break
    
    if integrated_path is None:
        logger.warning("Diretório de dados integrados não encontrado")
        return None, None
    
    # Tentar versões diferentes por ordem de preferência
    data_options = [
        ("X_integrated_real.npy", "y_integrated_real.npy", "real"),
        ("X_integrated_advanced.npy", "y_integrated_advanced.npy", "avançado"),
        ("X_integrated_simple.npy", "y_integrated_simple.npy", "simples"),
        ("X_integrated.npy", "y_integrated.npy", "básico")
    ]
    
    for x_file, y_file, version in data_options:
        X_path = integrated_path / x_file
        y_path = integrated_path / y_file
        
        if X_path.exists() and y_path.exists():
            logger.info(f"Carregando UNSW-NB15 {version} de: {integrated_path}")
            X = np.load(X_path)
            y = np.load(y_path)
            
            # Carregar metadados se disponível (para versão avançada)
            if version == "avançado":
                metadata_path = integrated_path / "metadata_advanced.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        logger.info(f"Features selecionadas: {metadata['n_features']}")
                        logger.info(f"Distribuição: {metadata['class_distribution']}")
            
            logger.info(f"Dataset {version} carregado: {X.shape}")
            return X, y
    
    logger.warning("Nenhum dado UNSW-NB15 encontrado!")
    return None, None

def train_advanced_hybrid():
    """Treina modelos híbridos com dados avançados"""
    logger.info("Iniciando treinamento híbrido avançado...")
    
    start_time = time.time()
    
    try:
        # Importar bibliotecas necessárias
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, IsolationForest
        from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
        import xgboost as xgb
        
        # 1. Carregar CIC-DDoS2019 
        logger.info("Carregando CIC-DDoS2019...")
        
        # Tentar diferentes localizações
        possible_paths = [
            Path("./src/datasets/cicddos2019"),
            Path("./datasets/cicddos2019"),
            Path("../src/datasets/cicddos2019"),
            Path("../../src/datasets/cicddos2019")
        ]
        
        cicddos_path = None
        for path in possible_paths:
            if path.exists():
                cicddos_path = path
                logger.info(f"CIC-DDoS2019 encontrado em: {path}")
                break
        
        if cicddos_path is None:
            raise FileNotFoundError("CIC-DDoS2019 não encontrado em nenhum local esperado")
        
        # Procurar pelo arquivo final combinado primeiro
        combined_file = cicddos_path / "cicddos2019_dataset.csv"
        if combined_file.exists():
            logger.info(f"Usando arquivo combinado: {combined_file}")
            
            # OTIMIZAÇÃO: Processar em chunks para economizar memória
            logger.info("Processando dataset em chunks para otimizar memória...")
            
            # Primeiro, ler apenas o cabeçalho para entender as colunas
            sample_df = pd.read_csv(combined_file, nrows=1000)
            
            # Identificar a coluna de label (pode ser 'Label', 'label', ou outro nome)
            label_col = None
            possible_labels = ['Label', 'label', 'class', 'Class', 'attack', 'Attack', 'type', 'Type']
            for col in possible_labels:
                if col in sample_df.columns:
                    label_col = col
                    break
            
            if label_col is None:
                logger.error("Nenhuma coluna de label encontrada")
                # Assumir que a última coluna é o label
                label_col = sample_df.columns[-1]
                logger.warning(f"Usando última coluna como label: {label_col}")
            
            logger.info(f"Usando coluna de label: {label_col}")
            
            columns_to_drop = ['Timestamp', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port']
            columns_to_drop = [col for col in columns_to_drop if col in sample_df.columns]
            
            # Ler o dataset em chunks menores (100k linhas por vez)
            chunk_size = 100000
            combined_chunks = []
            total_rows = 0
            max_rows = float('inf')  # Sem limite - processar todo o dataset
            
            logger.info(f"Processando em chunks de {chunk_size:,} linhas (sem limite)")
            
            # Coletar labels de todo o dataset primeiro
            all_labels = set()
            
            for chunk in pd.read_csv(combined_file, chunksize=chunk_size, low_memory=False):
                chunk = chunk.drop(columns=columns_to_drop, errors='ignore')
                chunk_labels = chunk[label_col].unique()
                all_labels.update(chunk_labels)
                
                # Balancear: manter proporção de ataques/benignos
                # Identificar valores benignos na coluna de label
                benign_values = ['BENIGN', 'Benign', 'benign', 'normal', 'Normal', 'NORMAL', '0']
                benign_mask = chunk[label_col].astype(str).isin(benign_values)
                attack_samples = chunk[~benign_mask]
                benign_samples = chunk[benign_mask]
                
                # Manter proporção 70% ataques, 30% benignos
                if len(attack_samples) > 0 and len(benign_samples) > 0:
                    attack_keep = min(len(attack_samples), int(chunk_size * 0.7))
                    benign_keep = min(len(benign_samples), int(chunk_size * 0.3))
                    
                    balanced_chunk = pd.concat([
                        attack_samples.sample(n=attack_keep, random_state=42),
                        benign_samples.sample(n=benign_keep, random_state=42)
                    ])
                    
                    combined_chunks.append(balanced_chunk)
                    total_rows += len(balanced_chunk)
                elif len(attack_samples) > 0:
                    # Só ataques neste chunk
                    combined_chunks.append(attack_samples)
                    total_rows += len(attack_samples)
                elif len(benign_samples) > 0:
                    # Só benignos neste chunk
                    combined_chunks.append(benign_samples)
                    total_rows += len(benign_samples)
                
                if total_rows % 100000 == 0:
                    logger.info(f"Processadas {total_rows:,} linhas...")
            
            logger.info(f"Labels únicos encontrados em todo o dataset: {sorted(list(all_labels))}")
            
            # Mostrar estatísticas dos tipos de ataques
            num_attack_types = len([label for label in all_labels if str(label) not in ['BENIGN', 'Benign', 'benign', 'normal', 'Normal', 'NORMAL', '0']])
            logger.info(f"Total de tipos de ataques únicos: {num_attack_types}")
            logger.info(f"Total de linhas processadas: {total_rows:,}")
            
            # Combinar chunks processados
            logger.info("Combinando chunks processados...")
            combined_df = pd.concat(combined_chunks, ignore_index=True)
            logger.info(f"Dataset processado: {combined_df.shape} (de ~29GB original)")
            
        else:
            # Fallback: carregar arquivos individuais
            csv_files = list(cicddos_path.glob("*.csv"))
            
            if not csv_files:
                raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {cicddos_path}")
            
            logger.info(f"Carregando {len(csv_files)} arquivos CSV individuais...")
            
            # Primeiro, coletar todos os labels únicos
            all_labels = set()
            dfs = []
            
            for i, file in enumerate(csv_files):
                logger.info(f"Processando arquivo {i+1}/{len(csv_files)}: {file.name}")
                df = pd.read_csv(file)
                
                # Coletar labels únicos deste arquivo
                if label_col in df.columns:
                    file_labels = df[label_col].unique()
                    all_labels.update(file_labels)
                    logger.info(f"  Labels em {file.name}: {sorted(list(file_labels))}")
                
                dfs.append(df)
            
            logger.info(f"Labels únicos encontrados em todos os arquivos: {sorted(list(all_labels))}")
            combined_df = pd.concat(dfs, ignore_index=True)
        
        # Preprocessar CIC-DDoS2019 (otimizado para memória)
        logger.info("Preprocessando dados...")
        
        # Identificar e remover colunas não-numéricas antes do processamento
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = combined_df.select_dtypes(exclude=[np.number]).columns
        
        # Manter apenas colunas numéricas e a coluna de label
        if len(non_numeric_cols) > 0:
            # Verificar se a coluna de label está nas não-numéricas (normal para strings como 'BENIGN')
            cols_to_keep = list(numeric_cols)
            if label_col not in numeric_cols:
                cols_to_keep.append(label_col)
            
            logger.info(f"Removendo {len(non_numeric_cols)} colunas não-numéricas (exceto label)")
            combined_df = combined_df[cols_to_keep]
        
        # Labels CIC-DDoS2019 - mapear diferentes formatos possíveis
        benign_values = ['BENIGN', 'Benign', 'benign', 'normal', 'Normal', 'NORMAL', '0']
        
        # Criar mapeamento de labels
        unique_labels = combined_df[label_col].unique()
        
        # Mapear para binário (0=benigno, 1=ataque)
        label_mapping = {}
        for label in unique_labels:
            if str(label) in benign_values:
                label_mapping[label] = 0
            else:
                label_mapping[label] = 1
        
        logger.info(f"Mapeamento de labels: {label_mapping}")
        y_cicddos = combined_df[label_col].map(label_mapping).fillna(1).values
        
        # Remover coluna de label para economizar memória
        X_cicddos = combined_df.drop(label_col, axis=1)
        del combined_df  # Liberar memória
        
        # Tratar valores infinitos e NaN de forma otimizada
        logger.info("Limpando dados...")
        
        # Processar apenas colunas numéricas
        numeric_cols = X_cicddos.select_dtypes(include=[np.number]).columns
        logger.info(f"Processando {len(numeric_cols)} colunas numéricas")
        
        for i, col in enumerate(numeric_cols):
            if i % 10 == 0:
                logger.info(f"Processando coluna {i+1}/{len(numeric_cols)}")
            
            # Converter para numérico e tratar infinitos/NaN
            X_cicddos[col] = pd.to_numeric(X_cicddos[col], errors='coerce')
            X_cicddos[col] = X_cicddos[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Converter para array numpy com tipo otimizado
        logger.info("Convertendo para numpy array...")
        
        # Filtrar apenas colunas numéricas
        numeric_cols = X_cicddos.select_dtypes(include=[np.number]).columns
        non_numeric_cols = X_cicddos.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_cols) > 0:
            logger.info(f"Removendo {len(non_numeric_cols)} colunas não-numéricas")
            X_cicddos = X_cicddos[numeric_cols]
        
        # Verificar se ainda temos colunas após filtragem
        if X_cicddos.shape[1] == 0:
            logger.error("Nenhuma coluna numérica encontrada após filtragem")
            return False
        
        X_cicddos = X_cicddos.astype(np.float32).values  # float32 usa metade da memória que float64
        y_cicddos = y_cicddos.astype(np.int32)  # int32 suficiente para labels
        
        logger.info(f"CIC-DDoS2019 processado: {X_cicddos.shape}, tipo: {X_cicddos.dtype}")
        logger.info(f"Uso de memória estimado: {X_cicddos.nbytes / (1024**3):.2f} GB")
        
        # Verificar se o dataset não está muito grande
        data_size_gb = X_cicddos.nbytes / (1024**3)
        logger.info(f"Uso de memória dos dados: {data_size_gb:.2f} GB")
        
        # Se dados muito grandes, usar versão simplificada automaticamente
        if data_size_gb > 4.0:  # Se dados > 4GB, usar modo lite
            logger.warning(f"Dataset muito grande ({data_size_gb:.2f}GB)")
            logger.info("Mudando automaticamente para modo lite...")
            
            # Executar script lite
            import subprocess
            script_dir = Path(__file__).parent
            lite_script = script_dir / "train_simple_lite.py"
            
            if lite_script.exists():
                result = subprocess.run([sys.executable, str(lite_script)], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("Treino lite concluído com sucesso")
                    return True
                else:
                    logger.error(f"Treino lite falhou: {result.stderr}")
                    return False
            else:
                logger.error("Script lite não encontrado")
                return False
        
        # Se dados > 2GB, reduzir mas continuar
        if data_size_gb > 2.0:
            logger.warning(f"Dataset grande ({data_size_gb:.2f}GB), reduzindo para otimizar memória")
            max_samples = min(len(X_cicddos), 200000)  # Máximo 200k amostras
            indices = np.random.choice(len(X_cicddos), max_samples, replace=False)
            X_cicddos = X_cicddos[indices]
            y_cicddos = y_cicddos[indices]
            logger.info(f"Dataset reduzido para: {X_cicddos.shape}")
        
        # 2. Treinar especialista DDoS (otimizado para memória)
        logger.info("Treinando especialista DDoS...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_cicddos, y_cicddos, test_size=0.2, random_state=42, stratify=y_cicddos
        )
        
        # Liberar memória original
        del X_cicddos, y_cicddos
        
        scaler_ddos = StandardScaler()
        X_train_scaled = scaler_ddos.fit_transform(X_train)
        X_test_scaled = scaler_ddos.transform(X_test)
        
        # XGBoost DDoS com configuração otimizada para memória
        xgb_ddos = xgb.XGBClassifier(
            n_estimators=100,  # Reduzido de 200 para 100
            max_depth=6,       # Reduzido de 8 para 6
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=1,          # Forçar single-thread para controlar memória
            tree_method='hist' # Método mais eficiente em memória
        )
        
        logger.info("Treinando XGBoost...")
        xgb_ddos.fit(X_train_scaled, y_train)
        
        xgb_ddos_pred = xgb_ddos.predict(X_test_scaled)
        xgb_ddos_f1 = f1_score(y_test, xgb_ddos_pred)
        xgb_ddos_precision = precision_score(y_test, xgb_ddos_pred)
        xgb_ddos_recall = recall_score(y_test, xgb_ddos_pred)
        
        logger.info(f"XGBoost DDoS - F1: {xgb_ddos_f1:.4f}, Precisão: {xgb_ddos_precision:.4f}, Recall: {xgb_ddos_recall:.4f}")
        
        # Random Forest DDoS (otimizado)
        rf_ddos = RandomForestClassifier(
            n_estimators=50,   # Reduzido de 150 para 50
            max_depth=10,      # Reduzido de 12 para 10
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1          # Single-thread para controlar memória
        )
        rf_ddos.fit(X_train_scaled, y_train)
        
        rf_ddos_pred = rf_ddos.predict(X_test_scaled)
        rf_ddos_f1 = f1_score(y_test, rf_ddos_pred)
        
        logger.info(f"Random Forest DDoS - F1: {rf_ddos_f1:.4f}")
        
        # 3. Carregar UNSW-NB15 avançado
        X_unsw, y_unsw = load_advanced_unsw_data()
        
        if X_unsw is not None:
            logger.info(f"UNSW-NB15 avançado carregado: {X_unsw.shape}")
            
            # Treinar detector generalista com dados avançados
            logger.info("Treinando detector generalista avançado...")
            
            X_train_unsw, X_test_unsw, y_train_unsw, y_test_unsw = train_test_split(
                X_unsw, y_unsw, test_size=0.2, random_state=42, stratify=y_unsw
            )
            
            # XGBoost Generalista com dados avançados
            xgb_general = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42
            )
            xgb_general.fit(X_train_unsw, y_train_unsw)
            
            xgb_general_pred = xgb_general.predict(X_test_unsw)
            xgb_general_f1 = f1_score(y_test_unsw, xgb_general_pred)
            xgb_general_precision = precision_score(y_test_unsw, xgb_general_pred)
            xgb_general_recall = recall_score(y_test_unsw, xgb_general_pred)
            
            logger.info(f"XGBoost Generalista - F1: {xgb_general_f1:.4f}, Precisão: {xgb_general_precision:.4f}, Recall: {xgb_general_recall:.4f}")
            
            # Isolation Forest melhorado
            normal_samples = X_train_unsw[y_train_unsw == 0]
            if_general = IsolationForest(
                n_estimators=150,
                contamination=0.2,
                random_state=42
            )
            if_general.fit(normal_samples)
            
            if_pred = if_general.predict(X_test_unsw)
            if_pred_binary = (if_pred == -1).astype(int)
            if_f1 = f1_score(y_test_unsw, if_pred_binary)
            
            logger.info(f"Isolation Forest - F1: {if_f1:.4f}")
            
            # 4. Salvar modelos melhorados
            models_path = Path("./models/hybrid_advanced")
            models_path.mkdir(parents=True, exist_ok=True)
            
            models_data = {
                'ddos_xgboost': xgb_ddos,
                'ddos_random_forest': rf_ddos,
                'ddos_scaler': scaler_ddos,
                'general_xgboost': xgb_general,
                'general_isolation_forest': if_general,
                'performance': {
                    'ddos_xgb_f1': xgb_ddos_f1,
                    'ddos_xgb_precision': xgb_ddos_precision,
                    'ddos_xgb_recall': xgb_ddos_recall,
                    'ddos_rf_f1': rf_ddos_f1,
                    'general_xgb_f1': xgb_general_f1,
                    'general_xgb_precision': xgb_general_precision,
                    'general_xgb_recall': xgb_general_recall,
                    'general_if_f1': if_f1
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(models_path / "hybrid_advanced_models.pkl", 'wb') as f:
                pickle.dump(models_data, f)
            
            with open(models_path / "performance_advanced.json", 'w') as f:
                json.dump(models_data['performance'], f, indent=2)
            
            # Estatísticas finais
            training_time = time.time() - start_time
            
            logger.info("Treinamento híbrido avançado concluído!")
            logger.info(f"Tempo total: {training_time:.2f}s")
            logger.info("RESULTADOS FINAIS:")
            logger.info(f"  Especialista DDoS XGB - F1: {xgb_ddos_f1:.4f} | Precisão: {xgb_ddos_precision:.4f} | Recall: {xgb_ddos_recall:.4f}")
            logger.info(f"  Especialista DDoS RF  - F1: {rf_ddos_f1:.4f}")
            logger.info(f"  Detector Geral XGB - F1: {xgb_general_f1:.4f} | Precisão: {xgb_general_precision:.4f} | Recall: {xgb_general_recall:.4f}")
            logger.info(f"  Detector Geral IF  - F1: {if_f1:.4f}")
            logger.info(f"Modelos salvos em: {models_path}")
            
        else:
            logger.error("UNSW-NB15 não disponível")
            
        return True
        
    except Exception as e:
        logger.error(f"Erro no treinamento avançado: {e}")
        return False

if __name__ == "__main__":
    success = train_advanced_hybrid()
    sys.exit(0 if success else 1)
