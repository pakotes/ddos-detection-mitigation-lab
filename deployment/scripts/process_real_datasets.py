#!/usr/bin/env python3
"""
Script SIMPLES para usar dados REAIS dos datasets
Carrega CIC-DDoS2019 e UNSW-NB15 originais e processa minimamente
"""

import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def load_real_cicddos_data():
    """Carregar dados REAIS do CIC-DDoS2019"""
    logger.info("Carregando dados REAIS CIC-DDoS2019...")
    
    # Procurar arquivos CSV do CIC-DDoS2019
    base_path = Path("./datasets/cicddos2019")
    
    if not base_path.exists():
        base_path = Path("../datasets/cicddos2019")
    
    if not base_path.exists():
        raise FileNotFoundError(f"Diretório CIC-DDoS2019 não encontrado. Coloque em: {base_path}")
    
    # Procurar arquivos CSV
    csv_files = list(base_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {base_path}")
    
    # Usar primeiro arquivo ou combinar vários
    if len(csv_files) == 1:
        logger.info(f"Usando arquivo: {csv_files[0].name}")
        df = pd.read_csv(csv_files[0])
    else:
        logger.info(f"Encontrados {len(csv_files)} arquivos, usando o maior...")
        # Usar o arquivo maior
        largest_file = max(csv_files, key=lambda f: f.stat().st_size)
        logger.info(f"Usando arquivo: {largest_file.name}")
        df = pd.read_csv(largest_file)
    
    logger.info(f"Dataset CIC-DDoS carregado: {df.shape}")
    return df

def load_real_unsw_data():
    """Carregar dados REAIS do UNSW-NB15"""
    logger.info("Carregando dados REAIS UNSW-NB15...")
    
    # Procurar arquivos do UNSW-NB15
    base_path = Path("./datasets/unsw-nb15")
    
    if not base_path.exists():
        base_path = Path("../datasets/unsw-nb15")
    
    if not base_path.exists():
        raise FileNotFoundError(f"Diretório UNSW-NB15 não encontrado. Coloque em: {base_path}")
    
    # Procurar arquivo principal
    main_file = base_path / "NF-UNSW-NB15-v3.csv"
    
    if not main_file.exists():
        # Procurar outros arquivos CSV
        csv_files = list(base_path.glob("*.csv"))
        csv_files = [f for f in csv_files if "feature" not in f.name.lower()]
        
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo de dados encontrado em {base_path}")
        
        main_file = csv_files[0]
    
    logger.info(f"Usando arquivo: {main_file.name}")
    df = pd.read_csv(main_file)
    
    logger.info(f"Dataset UNSW carregado: {df.shape}")
    return df

def process_real_cicddos(df, max_samples=50000):
    """Processar dados REAIS do CIC-DDoS2019"""
    logger.info("Processando dados REAIS CIC-DDoS2019...")
    
    # Detectar coluna de label automaticamente
    label_col = None
    for col in ['Label', 'label', 'Label ', ' Label']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        # Procurar por padrões comuns
        for col in df.columns:
            if 'label' in col.lower() or col.lower() in ['attack', 'class', 'type']:
                label_col = col
                break
    
    if label_col is None:
        raise ValueError(f"Coluna de label não encontrada. Colunas disponíveis: {list(df.columns)}")
    
    logger.info(f"Usando coluna de label: {label_col}")
    
    # Separar features e labels
    y = df[label_col].astype(str).str.strip()
    X = df.drop(columns=[label_col])
    
    # Remover colunas não numéricas ou problemáticas
    non_numeric_cols = []
    for col in X.columns:
        if X[col].dtype == 'object':
            non_numeric_cols.append(col)
        elif 'ip' in col.lower() or 'addr' in col.lower():
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        logger.info(f"Removendo colunas não numéricas: {len(non_numeric_cols)} colunas")
        X = X.drop(columns=non_numeric_cols)
    
    # Limpar dados
    logger.info("Limpando dados...")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Converter para numérico
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    # Limitar tamanho se muito grande
    if len(X) > max_samples:
        logger.info(f"Dataset muito grande ({len(X)}), amostrando {max_samples}...")
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X.iloc[indices]
        y = y.iloc[indices]
    
    # Verificar distribuição de classes
    class_counts = y.value_counts()
    logger.info("Distribuição de classes:")
    for class_name, count in class_counts.items():
        logger.info(f"  {class_name}: {count:,} ({count/len(y):.1%})")
    
    return X.values, y.values

def process_real_unsw(df, max_samples=30000):
    """Processar dados REAIS do UNSW-NB15"""
    logger.info("Processando dados REAIS UNSW-NB15...")
    
    # Detectar coluna de label
    label_col = None
    for col in ['Label', 'label', 'attack_cat', 'Attack']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"Coluna de label não encontrada. Colunas disponíveis: {list(df.columns)}")
    
    logger.info(f"Usando coluna de label: {label_col}")
    
    # Separar features e labels
    y = df[label_col].astype(str).str.strip()
    X = df.drop(columns=[label_col])
    
    # Remover colunas problemáticas
    cols_to_remove = []
    for col in X.columns:
        if X[col].dtype == 'object':
            cols_to_remove.append(col)
        elif 'ip' in col.lower() or 'addr' in col.lower():
            cols_to_remove.append(col)
    
    if cols_to_remove:
        logger.info(f"Removendo colunas problemáticas: {len(cols_to_remove)} colunas")
        X = X.drop(columns=cols_to_remove)
    
    # Limpar dados
    logger.info("Limpando dados...")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Converter para numérico
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    # Limitar tamanho
    if len(X) > max_samples:
        logger.info(f"Dataset muito grande ({len(X)}), amostrando {max_samples}...")
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X.iloc[indices]
        y = y.iloc[indices]
    
    # Verificar distribuição
    class_counts = y.value_counts()
    logger.info("Distribuição de classes:")
    for class_name, count in class_counts.items():
        logger.info(f"  {class_name}: {count:,} ({count/len(y):.1%})")
    
    return X.values, y.values

def save_real_dataset(X, y, dataset_name, output_dir):
    """Salvar dataset real processado"""
    logger.info(f"Salvando dataset real: {dataset_name}...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Codificar labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Labels binários (0=normal, 1=ataque)
    if dataset_name == 'cicddos':
        normal_labels = ['BENIGN', 'Benign', 'benign']
    else:
        normal_labels = ['Normal', 'normal', '0']
    
    y_binary = np.array([
        0 if any(norm in str(label) for norm in normal_labels) else 1 
        for label in y
    ])
    
    # Normalização simples
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Seleção de features (top 50)
    logger.info("Selecionando features mais importantes...")
    selector = SelectKBest(score_func=f_classif, k=min(50, X.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y_binary)
    
    # Split
    X_train, X_test, y_train_multi, y_test_multi, y_train_bin, y_test_bin = train_test_split(
        X_selected, y_encoded, y_binary, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_binary
    )
    
    # Salvar
    np.save(output_dir / f'X_train_{dataset_name}.npy', X_train)
    np.save(output_dir / f'X_test_{dataset_name}.npy', X_test)
    np.save(output_dir / f'y_train_multi_{dataset_name}.npy', y_train_multi)
    np.save(output_dir / f'y_test_multi_{dataset_name}.npy', y_test_multi)
    np.save(output_dir / f'y_train_binary_{dataset_name}.npy', y_train_bin)
    np.save(output_dir / f'y_test_binary_{dataset_name}.npy', y_test_bin)
    
    # Metadados
    metadata = {
        'dataset_name': dataset_name,
        'source': 'REAL_DATA',
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X_selected.shape[1],
        'n_classes': len(np.unique(y_encoded)),
        'attack_ratio': float(y_binary.mean()),
        'normal_ratio': float((y_binary == 0).mean()),
        'classes': le.classes_.tolist(),
        'processing': [
            'Real data from original datasets',
            'Automatic label detection',
            'Cleaned infinite values and NaN',
            'Feature selection (top 50)',
            'Standard scaling'
        ]
    }
    
    with open(output_dir / f'metadata_{dataset_name}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Salvar processadores
    with open(output_dir / f'scaler_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(output_dir / f'selector_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(selector, f)
    
    with open(output_dir / f'label_encoder_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    logger.info(f"Dataset real {dataset_name} salvo:")
    logger.info(f"  Treino: {X_train.shape}")
    logger.info(f"  Teste: {X_test.shape}")
    logger.info(f"  Taxa de ataques: {y_binary.mean():.1%}")
    
    return metadata

def main():
    """Função principal - SIMPLES"""
    logger.info("Processando dados REAIS dos datasets originais...")
    
    output_dir = Path(__file__).parent.parent.parent / 'src' / 'datasets' / 'real'
    
    try:
        # 1. CIC-DDoS2019 REAL
        logger.info("\n1️⃣ Processando CIC-DDoS2019 REAL...")
        try:
            df_cic = load_real_cicddos_data()
            X_cic, y_cic = process_real_cicddos(df_cic)
            metadata_cic = save_real_dataset(X_cic, y_cic, 'cicddos', output_dir)
            logger.info("✅ CIC-DDoS2019 processado com sucesso!")
        except Exception as e:
            logger.error(f"❌ Erro no CIC-DDoS2019: {e}")
            logger.info("Continuando sem este dataset...")
            metadata_cic = None
        
        # 2. UNSW-NB15 REAL
        logger.info("\n2️⃣ Processando UNSW-NB15 REAL...")
        try:
            df_unsw = load_real_unsw_data()
            X_unsw, y_unsw = process_real_unsw(df_unsw)
            metadata_unsw = save_real_dataset(X_unsw, y_unsw, 'unsw', output_dir)
            logger.info("✅ UNSW-NB15 processado com sucesso!")
        except Exception as e:
            logger.error(f"❌ Erro no UNSW-NB15: {e}")
            logger.info("Continuando sem este dataset...")
            metadata_unsw = None
        
        # Resumo
        summary = {
            'datasets_processed': [],
            'source': 'REAL_ORIGINAL_DATA',
            'processing_date': pd.Timestamp.now().isoformat(),
            'description': 'Dados reais dos datasets originais, processamento mínimo'
        }
        
        if metadata_cic:
            summary['datasets_processed'].append('cicddos')
            summary['cicddos'] = metadata_cic
        
        if metadata_unsw:
            summary['datasets_processed'].append('unsw')
            summary['unsw'] = metadata_unsw
        
        with open(output_dir / 'summary_real.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✅ Processamento concluído!")
        logger.info(f"Datasets processados: {len(summary['datasets_processed'])}")
        logger.info(f"Localização: {output_dir}")
        logger.info(f"📊 Use 'ddos-train-real' para treinar com dados reais")
        
        return len(summary['datasets_processed']) > 0
        
    except Exception as e:
        logger.error(f"Erro geral: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
