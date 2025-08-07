#!/usr/bin/env python3
"""
Feature Engineering Avançado para UNSW-NB15
Criação de features temporais e estatísticas avançadas
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_raw_unsw_data():
    """Carrega dados raw do UNSW-NB15"""
    logger.info("Carregando dados raw UNSW-NB15...")
    
    data_path = Path("./datasets/unsw-nb15/NF-UNSW-NB15-v3.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Dataset carregado: {df.shape}")
    
    return df

def create_advanced_temporal_features(df):
    """Cria features temporais avançadas"""
    logger.info("⏰ Criando features temporais avançadas...")
    
    # Features de tempo absoluto
    df['flow_start_hour'] = (df['FLOW_START_MILLISECONDS'] // (1000 * 3600)) % 24
    df['flow_start_day_of_week'] = (df['FLOW_START_MILLISECONDS'] // (1000 * 3600 * 24)) % 7
    
    # Features de duração
    df['duration_ratio'] = df['DURATION_OUT'] / (df['DURATION_IN'] + 1e-8)
    df['flow_efficiency'] = df['FLOW_DURATION_MILLISECONDS'] / (df['IN_PKTS'] + df['OUT_PKTS'] + 1e-8)
    
    # Features de inter-arrival time derivadas
    df['iat_variability'] = df['SRC_TO_DST_IAT_STDDEV'] / (df['SRC_TO_DST_IAT_AVG'] + 1e-8)
    df['iat_asymmetry'] = df['SRC_TO_DST_IAT_AVG'] / (df['DST_TO_SRC_IAT_AVG'] + 1e-8)
    
    # Features de burst detection
    df['burst_indicator'] = (df['SRC_TO_DST_IAT_MIN'] < 10) & (df['SRC_TO_DST_IAT_MAX'] > 1000)
    
    logger.info("Features temporais criadas")
    return df

def create_advanced_traffic_features(df):
    """Cria features de tráfego avançadas"""
    logger.info("Criando features de tráfego avançadas...")
    
    # Features de bytes
    df['bytes_ratio'] = df['OUT_BYTES'] / (df['IN_BYTES'] + 1e-8)
    df['bytes_per_packet_in'] = df['IN_BYTES'] / (df['IN_PKTS'] + 1e-8)
    df['bytes_per_packet_out'] = df['OUT_BYTES'] / (df['OUT_PKTS'] + 1e-8)
    df['bytes_asymmetry'] = abs(df['IN_BYTES'] - df['OUT_BYTES']) / (df['IN_BYTES'] + df['OUT_BYTES'] + 1e-8)
    
    # Features de throughput derivadas
    df['throughput_ratio'] = df['DST_TO_SRC_AVG_THROUGHPUT'] / (df['SRC_TO_DST_AVG_THROUGHPUT'] + 1e-8)
    df['throughput_stability'] = df['SRC_TO_DST_AVG_THROUGHPUT'] / (df['SRC_TO_DST_SECOND_BYTES'] + 1e-8)
    
    # Features de distribuição de pacotes
    total_pkts = (df['NUM_PKTS_UP_TO_128_BYTES'] + df['NUM_PKTS_128_TO_256_BYTES'] + 
                  df['NUM_PKTS_256_TO_512_BYTES'] + df['NUM_PKTS_512_TO_1024_BYTES'] + 
                  df['NUM_PKTS_1024_TO_1514_BYTES'])
    
    df['small_packets_ratio'] = df['NUM_PKTS_UP_TO_128_BYTES'] / (total_pkts + 1e-8)
    df['large_packets_ratio'] = df['NUM_PKTS_1024_TO_1514_BYTES'] / (total_pkts + 1e-8)
    df['packet_size_diversity'] = (df['MAX_IP_PKT_LEN'] - df['MIN_IP_PKT_LEN']) / (df['MAX_IP_PKT_LEN'] + 1e-8)
    
    # Features de retransmissão
    df['retrans_ratio'] = (df['RETRANSMITTED_IN_BYTES'] + df['RETRANSMITTED_OUT_BYTES']) / (df['IN_BYTES'] + df['OUT_BYTES'] + 1e-8)
    df['retrans_efficiency'] = df['RETRANSMITTED_IN_PKTS'] / (df['RETRANSMITTED_IN_BYTES'] + 1e-8)
    
    logger.info("Features de tráfego criadas")
    return df

def create_protocol_features(df):
    """Cria features baseadas em protocolos"""
    logger.info("Criando features de protocolo...")
    
    # Features TCP - corrigir o processamento de TCP flags
    def count_tcp_flags(x):
        try:
            if pd.isna(x) or x == 0:
                return 0
            return bin(int(x)).count('1')
        except:
            return 0
    
    df['tcp_flag_diversity'] = df['TCP_FLAGS'].apply(count_tcp_flags)
    df['tcp_window_efficiency'] = df['TCP_WIN_MAX_OUT'] / (df['TCP_WIN_MAX_IN'] + 1e-8)
    
    # Features específicas por protocolo
    df['is_tcp'] = (df['PROTOCOL'] == 6).astype(int)
    df['is_udp'] = (df['PROTOCOL'] == 17).astype(int)
    df['is_icmp'] = (df['PROTOCOL'] == 1).astype(int)
    
    # Features de aplicação
    df['has_dns'] = (df['DNS_QUERY_ID'] > 0).astype(int)
    df['has_ftp'] = (df['FTP_COMMAND_RET_CODE'] > 0).astype(int)
    
    logger.info("Features de protocolo criadas")
    return df

def create_anomaly_detection_features(df):
    """Cria features específicas para detecção de anomalias"""
    logger.info("Criando features de detecção de anomalias...")
    
    # Z-scores para detecção de outliers
    numeric_cols = ['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']
    
    for col in numeric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            df[f'{col}_zscore'] = abs((df[col] - mean_val) / (std_val + 1e-8))
            df[f'{col}_is_outlier'] = (df[f'{col}_zscore'] > 3).astype(int)
    
    # Features de comportamento suspeito
    df['suspicious_port'] = ((df['L4_SRC_PORT'] < 1024) | (df['L4_DST_PORT'] < 1024)).astype(int)
    df['high_throughput'] = (df['SRC_TO_DST_AVG_THROUGHPUT'] > df['SRC_TO_DST_AVG_THROUGHPUT'].quantile(0.95)).astype(int)
    df['unusual_packet_size'] = ((df['MIN_IP_PKT_LEN'] < 40) | (df['MAX_IP_PKT_LEN'] > 1500)).astype(int)
    
    logger.info("Features de anomalia criadas")
    return df

def advanced_feature_selection(X, y, n_features=100):
    """Seleção avançada de features"""
    logger.info(f"Seleção avançada de features (top {n_features})...")
    
    # Múltiplos métodos de seleção
    selectors = {
        'f_classif': SelectKBest(score_func=f_classif, k=n_features),
        'mutual_info': SelectKBest(score_func=mutual_info_classif, k=n_features)
    }
    
    selected_features = set()
    
    for name, selector in selectors.items():
        logger.info(f"Aplicando {name}...")
        selector.fit(X, y)
        selected_idx = selector.get_support(indices=True)
        selected_features.update(selected_idx)
        logger.info(f"{name}: {len(selected_idx)} features selecionadas")
    
    # Combinar features selecionadas
    final_features = sorted(list(selected_features))[:n_features]
    X_selected = X.iloc[:, final_features]
    
    logger.info(f"{len(final_features)} features finais selecionadas")
    return X_selected, final_features

def robust_preprocessing(X):
    """Preprocessing robusto para melhor performance"""
    logger.info("Aplicando preprocessing robusto...")
    
    # Usar RobustScaler para lidar melhor com outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Transformação quantile para normalizar distribuições
    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=42)
    X_normalized = quantile_transformer.fit_transform(X_scaled)
    
    logger.info("Preprocessing concluído")
    return X_normalized, scaler, quantile_transformer

def save_advanced_features(X, y, feature_names, scalers, output_dir="./datasets/integrated"):
    """Salva features avançadas"""
    logger.info("💾 Salvando features avançadas...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Salvar dados
    np.save(output_path / "X_integrated_advanced.npy", X)
    np.save(output_path / "y_integrated_advanced.npy", y)
    
    # Salvar metadados
    metadata = {
        'source': 'UNSW-NB15 NetFlow v3 - Advanced Feature Engineering',
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'class_distribution': {
            'normal': int(np.sum(y == 0)),
            'attack': int(np.sum(y == 1))
        },
        'selected_features': feature_names,
        'preprocessing': [
            'advanced_temporal_features',
            'advanced_traffic_features', 
            'protocol_features',
            'anomaly_detection_features',
            'robust_scaling',
            'quantile_normalization',
            'advanced_feature_selection'
        ]
    }
    
    with open(output_path / "metadata_advanced.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Salvar scalers
    import pickle
    with open(output_path / "scalers_advanced.pkl", 'wb') as f:
        pickle.dump(scalers, f)
    
    logger.info(f"Features avançadas salvas em: {output_path}")
    return metadata

def main():
    """Função principal"""
    try:
        logger.info("Iniciando feature engineering avançado...")
        
        # 1. Carregar dados raw
        df = load_raw_unsw_data()
        
        # 2. Criar features avançadas
        df = create_advanced_temporal_features(df)
        df = create_advanced_traffic_features(df)
        df = create_protocol_features(df)
        df = create_anomaly_detection_features(df)
        
        # 3. Preparar dados
        # Remover colunas não necessárias
        drop_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack']
        df_processed = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # Separar features e labels
        y = df_processed['Label'].values
        X = df_processed.drop('Label', axis=1)
        
        # Limpar dados
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Converter para numérico
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        logger.info(f"Dataset processado: {X.shape}")
        
        # 4. Seleção avançada de features
        X_selected, selected_features = advanced_feature_selection(X, y, n_features=100)
        
        # 5. Preprocessing robusto
        X_preprocessed, scaler, quantile_transformer = robust_preprocessing(X_selected)
        
        # 6. Balanceamento (usar 300k amostras para mais diversidade)
        n_samples = min(300000, len(X_preprocessed))
        indices = np.random.choice(len(X_preprocessed), n_samples, replace=False)
        X_final = X_preprocessed[indices]
        y_final = y[indices]
        
        # 7. Salvar resultados
        feature_names = [X_selected.columns[i] for i in range(len(selected_features))]
        scalers = {'robust_scaler': scaler, 'quantile_transformer': quantile_transformer}
        
        metadata = save_advanced_features(X_final, y_final, feature_names, scalers)
        
        logger.info("Feature engineering avançado concluído!")
        logger.info(f"Shape final: {X_final.shape}")
        logger.info(f"Distribuição: Normal={metadata['class_distribution']['normal']}, Attack={metadata['class_distribution']['attack']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro no feature engineering: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
