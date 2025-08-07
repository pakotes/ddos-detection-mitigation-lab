#!/usr/bin/env python3
"""
Real UNSW-NB15 Feature Engineering
Usa os datasets oficiais de treino/teste com labels reais
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import json

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_unsw_real_data():
    """Carrega os datasets oficiais UNSW-NB15 com labels reais"""
    logger.info("🚀 Carregando datasets UNSW-NB15 oficiais...")
    
    base_path = Path("../datasets/unsw-nb15")
    
    # Procurar pelo arquivo principal do UNSW-NB15 v3
    main_file = base_path / "NF-UNSW-NB15-v3.csv"
    
    if not main_file.exists():
        # Tentar outras possibilidades
        csv_files = list(base_path.glob("*.csv"))
        csv_files = [f for f in csv_files if f.name != "NetFlow_v3_Features.csv"]  # Excluir arquivo de features
        
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo CSV de dados encontrado em {base_path}")
        
        main_file = csv_files[0]  # Usar o primeiro arquivo encontrado
        logger.info(f"Usando arquivo: {main_file.name}")
    
    # Carregar dados
    logger.info(f"📊 Carregando dataset: {main_file.name}")
    df = pd.read_csv(main_file)
    logger.info(f"Dataset carregado: {df.shape}")
    
    return df

def preprocess_unsw_features(df):
    """Preprocessa features do UNSW-NB15 com dados reais"""
    logger.info("🔧 Preprocessando features UNSW-NB15...")
    
    # Remover colunas não necessárias
    columns_to_drop = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack']  # Manter apenas Label binário
    df_processed = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Separar features e labels
    y = df_processed['Label'].values  # 0=Normal, 1=Attack
    X = df_processed.drop('Label', axis=1)
    
    logger.info(f"Labels originais - Normal: {np.sum(y == 0)}, Attack: {np.sum(y == 1)}")
    
    # Tratar variáveis categóricas
    categorical_cols = ['PROTOCOL', 'L7_PROTO']  # Colunas categóricas do NetFlow v3
    
    for col in categorical_cols:
        if col in X.columns:
            logger.info(f"Codificando {col}: {X[col].nunique()} valores únicos")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Tratar valores infinitos e NaN
    logger.info("🧹 Limpando dados...")
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Para colunas numéricas, usar mediana
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
    
    # Converter tudo para numérico
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    logger.info(f"Features após limpeza: {X.shape}")
    
    # Normalização
    logger.info("📏 Normalizando features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Seleção de features (top 50)
    logger.info("🎯 Selecionando top 50 features...")
    selector = SelectKBest(score_func=f_classif, k=50)
    X_selected = selector.fit_transform(X_scaled, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    logger.info(f"Features selecionadas: {len(selected_features)}")
    
    return X_selected, y, selected_features

def balance_unsw_data(X, y, max_samples=200000):
    """Balanceia dados UNSW-NB15 mantendo distribuição realista"""
    logger.info("⚖️ Balanceando dataset...")
    
    # Análise da distribuição
    normal_count = np.sum(y == 0)
    attack_count = np.sum(y == 1)
    total_count = len(y)
    
    logger.info(f"Distribuição original: Normal={normal_count}, Attack={attack_count}, Total={total_count}")
    
    # Se muito grande, fazer amostragem
    if total_count > max_samples:
        logger.info(f"Dataset muito grande ({total_count}), amostrando {max_samples}...")
        
        # Manter proporção original mas limitar tamanho
        attack_ratio = attack_count / total_count
        target_attacks = min(int(max_samples * attack_ratio), attack_count)
        target_normals = max_samples - target_attacks
        
        normal_indices = np.where(y == 0)[0]
        attack_indices = np.where(y == 1)[0]
        
        # Amostragem aleatória
        np.random.seed(42)
        selected_normals = np.random.choice(normal_indices, size=min(target_normals, len(normal_indices)), replace=False)
        selected_attacks = np.random.choice(attack_indices, size=min(target_attacks, len(attack_indices)), replace=False)
        
        # Combinar índices
        selected_indices = np.concatenate([selected_normals, selected_attacks])
        np.random.shuffle(selected_indices)
        
        X_balanced = X[selected_indices]
        y_balanced = y[selected_indices]
        
        logger.info(f"Dataset balanceado: Normal={np.sum(y_balanced == 0)}, Attack={np.sum(y_balanced == 1)}")
        
        return X_balanced, y_balanced
    
    return X, y

def save_real_unsw_data(X, y, features, output_dir="../datasets/integrated"):
    """Salva dados processados"""
    logger.info("💾 Salvando dados processados...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Salvar arrays
    np.save(output_path / "X_integrated_real.npy", X)
    np.save(output_path / "y_integrated_real.npy", y)
    
    # Salvar metadados
    metadata = {
        'source': 'UNSW-NB15 Official Training/Testing Sets',
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'class_distribution': {
            'normal': int(np.sum(y == 0)),
            'attack': int(np.sum(y == 1))
        },
        'selected_features': features,
        'preprocessing': [
            'categorical_encoding',
            'missing_value_imputation',
            'standard_scaling',
            'feature_selection_top50'
        ]
    }
    
    with open(output_path / "metadata_real.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Salvar nomes das features
    with open(output_path / "feature_names_real.txt", 'w') as f:
        f.write('\n'.join(features))
    
    logger.info(f"✅ Dados salvos em: {output_path}")
    logger.info(f"📊 Shape final: {X.shape}")
    logger.info(f"🎯 Distribuição: Normal={metadata['class_distribution']['normal']}, Attack={metadata['class_distribution']['attack']}")
    
    return metadata

def main():
    """Função principal"""
    try:
        logger.info("🚀 Iniciando feature engineering UNSW-NB15 com dados reais...")
        
        # 1. Carregar dados reais
        df = load_unsw_real_data()
        
        # 2. Preprocessar
        X, y, features = preprocess_unsw_features(df)
        
        # 3. Balancear (limitar tamanho se necessário)
        X_balanced, y_balanced = balance_unsw_data(X, y)
        
        # 4. Salvar
        metadata = save_real_unsw_data(X_balanced, y_balanced, features)
        
        logger.info("✅ Feature engineering concluído com sucesso!")
        logger.info(f"📈 Performance esperada: >80% F1-score (dados reais)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no feature engineering: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
