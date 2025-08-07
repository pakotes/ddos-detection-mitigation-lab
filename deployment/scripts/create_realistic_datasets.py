#!/usr/bin/env python3
"""
Script para criar datasets limpos e mais realistas
Gera versões pequenas mas desafiador    """Criar amostra realista do NF-UNSW-NB15-v3 com mais desafios"""
    logger.info("Criando dataset realista NF-UNSW-NB15-v3...")
    
    np.random.seed(42)
    
    # Tipos de ataques NF-UNSW-NB15-v3 datasets originais
"""

import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def create_realistic_cicddos_sample():
    """Criar amostra realista do CIC-DDoS2019 com mais desafios"""
    logger.info("Criando dataset realista CIC-DDoS2019...")
    
    np.random.seed(42)
    
    # Tipos de ataques CIC-DDoS2019
    attack_types = [
        'BENIGN',
        'UDP-lag', 'WebDDoS', 'TFTP',
        'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL',
        'DrDoS_NetBIOS', 'DrDoS_NTP', 'DrDoS_SNMP',
        'DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'UDPLag'
    ]
    
    # Distribuição mais realista: 70% ataques, 30% benignos
    total_samples = 50000
    benign_samples = int(total_samples * 0.3)  # 30% benignos
    attack_samples_per_type = (total_samples - benign_samples) // (len(attack_types) - 1)
    
    data = []
    labels = []
    
    for i, attack_type in enumerate(attack_types):
        if attack_type == 'BENIGN':
            n_samples = benign_samples
        else:
            n_samples = attack_samples_per_type
        
        logger.info(f"Gerando {n_samples:,} amostras para {attack_type}...")
        
        n_features = 80  # Similar ao CIC-DDoS2019 real
        
        if attack_type == 'BENIGN':
            # Tráfego normal - valores baixos com alguma variação
            features = np.random.normal(0.1, 0.08, (n_samples, n_features))
            features = np.abs(features)
            
            # Adicionar ruído para tornar mais desafiador
            noise = np.random.normal(0, 0.02, features.shape)
            features += noise
            
        else:
            # Tráfego de ataque - valores mais altos mas com overlap com benignos
            base = 0.3 + i * 0.05  # Cada tipo tem padrão ligeiramente diferente
            features = np.random.normal(base, 0.15, (n_samples, n_features))
            features = np.abs(features)
            
            # Adicionar overlap intencional com tráfego benigno (torna mais desafiador)
            overlap_mask = np.random.random(n_samples) < 0.15  # 15% de overlap
            features[overlap_mask] = np.random.normal(0.15, 0.1, (overlap_mask.sum(), n_features))
            features[overlap_mask] = np.abs(features[overlap_mask])
            
            # Padrões específicos por tipo de ataque
            if 'DrDoS' in attack_type:
                features[:, :10] *= np.random.uniform(1.5, 2.5, (n_samples, 10))
            elif 'UDP' in attack_type:
                features[:, 10:20] *= np.random.uniform(2.0, 3.0, (n_samples, 10))
            elif 'Web' in attack_type:
                features[:, 20:30] *= np.random.uniform(1.8, 2.8, (n_samples, 10))
            elif 'Syn' in attack_type:
                features[:, 30:40] *= np.random.uniform(2.2, 3.2, (n_samples, 10))
            
            # Adicionar ruído realista
            noise = np.random.normal(0, 0.05, features.shape)
            features += noise
        
        data.append(features)
        labels.extend([attack_type] * n_samples)
    
    # Combinar todos os dados
    X = np.vstack(data)
    y = np.array(labels)
    
    # Embaralhar dados
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Verificar distribuição final
    unique, counts = np.unique(y, return_counts=True)
    attack_count = len(y) - counts[unique == 'BENIGN'][0]
    benign_count = counts[unique == 'BENIGN'][0]
    
    logger.info(f"Dataset criado: {X.shape} amostras")
    logger.info(f"Benignos: {benign_count:,} ({benign_count/len(y):.1%})")
    logger.info(f"Ataques: {attack_count:,} ({attack_count/len(y):.1%})")
    logger.info(f"Classes: {len(unique)}")
    
    return X, y, attack_types

def create_realistic_unsw_sample():
    """Criar amostra realista do NF-UNSW-NB15-v3 com mais desafios"""
    logger.info("Criando dataset realista NF-UNSW-NB15-v3...")
    
    np.random.seed(123)
    
    # Tipos de ataques NF-UNSW-NB15-v3
    attack_types = [
        'Normal',
        'Generic', 'Exploits', 'Fuzzers', 'DoS',
        'Reconnaissance', 'Analysis', 'Backdoor',
        'Shellcode', 'Worms'
    ]
    
    # Distribuição mais realista: 60% ataques, 40% normais
    total_samples = 30000
    normal_samples = int(total_samples * 0.4)  # 40% normais
    attack_samples_per_type = (total_samples - normal_samples) // (len(attack_types) - 1)
    
    data = []
    labels = []
    
    for i, attack_type in enumerate(attack_types):
        if attack_type == 'Normal':
            n_samples = normal_samples
        else:
            n_samples = attack_samples_per_type
        
        logger.info(f"Gerando {n_samples:,} amostras para {attack_type}...")
        
        n_features = 42  # Similar ao NF-UNSW-NB15-v3 real
        
        if attack_type == 'Normal':
            # Tráfego normal com variação realista
            features = np.random.normal(0.2, 0.12, (n_samples, n_features))
            features = np.abs(features)
            
            # Adicionar ruído
            noise = np.random.normal(0, 0.03, features.shape)
            features += noise
            
        else:
            # Tráfego de ataque com overlap intencional
            base = 0.25 + i * 0.06
            features = np.random.normal(base, 0.18, (n_samples, n_features))
            features = np.abs(features)
            
            # Overlap com tráfego normal (torna mais desafiador)
            overlap_mask = np.random.random(n_samples) < 0.20  # 20% de overlap
            features[overlap_mask] = np.random.normal(0.25, 0.12, (overlap_mask.sum(), n_features))
            features[overlap_mask] = np.abs(features[overlap_mask])
            
            # Padrões específicos
            if attack_type == 'DoS':
                features[:, :5] *= np.random.uniform(2.5, 4.0, (n_samples, 5))
            elif attack_type == 'Reconnaissance':
                features[:, 5:10] *= np.random.uniform(1.5, 2.5, (n_samples, 5))
            elif attack_type == 'Exploits':
                features[:, 10:15] *= np.random.uniform(2.0, 3.5, (n_samples, 5))
            elif attack_type == 'Fuzzers':
                features[:, 15:20] *= np.random.uniform(1.8, 3.0, (n_samples, 5))
            
            # Adicionar ruído realista
            noise = np.random.normal(0, 0.06, features.shape)
            features += noise
        
        data.append(features)
        labels.extend([attack_type] * n_samples)
    
    # Combinar todos os dados
    X = np.vstack(data)
    y = np.array(labels)
    
    # Embaralhar dados
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Verificar distribuição final
    unique, counts = np.unique(y, return_counts=True)
    normal_count = counts[unique == 'Normal'][0]
    attack_count = len(y) - normal_count
    
    logger.info(f"Dataset criado: {X.shape} amostras")
    logger.info(f"Normais: {normal_count:,} ({normal_count/len(y):.1%})")
    logger.info(f"Ataques: {attack_count:,} ({attack_count/len(y):.1%})")
    logger.info(f"Classes: {len(unique)}")
    
    return X, y, attack_types

def preprocess_and_save(X, y, dataset_name, output_dir):
    """Preprocessar e salvar dataset com balanceamento garantido"""
    logger.info(f"Preprocessando dataset {dataset_name}...")
    
    # Criar diretório de saída
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Codificar labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Criar labels binários (0=normal, 1=ataque)
    if dataset_name == 'cicddos':
        normal_labels = ['BENIGN']
    else:  # unsw
        normal_labels = ['Normal']
    
    y_binary = np.array([0 if label in normal_labels else 1 for label in y])
    
    # Verificar balanceamento antes do split
    normal_ratio = (y_binary == 0).mean()
    attack_ratio = (y_binary == 1).mean()
    
    logger.info(f"Distribuição final: {normal_ratio:.1%} normais, {attack_ratio:.1%} ataques")
    
    if normal_ratio < 0.2 or attack_ratio < 0.2:
        logger.warning(f"Dataset desequilibrado: {normal_ratio:.1%} normais")
    
    # Normalizar features (com clipping para evitar outliers extremos)
    X_clipped = np.clip(X, np.percentile(X, 1), np.percentile(X, 99))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clipped.astype(np.float32))
    
    # Split treino/teste estratificado
    X_train, X_test, y_train_multi, y_test_multi, y_train_bin, y_test_bin = train_test_split(
        X_scaled, y_encoded, y_binary, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_binary
    )
    
    # Verificar balanceamento após split
    train_normal_ratio = (y_train_bin == 0).mean()
    test_normal_ratio = (y_test_bin == 0).mean()
    
    logger.info(f"Treino: {train_normal_ratio:.1%} normais")
    logger.info(f"Teste: {test_normal_ratio:.1%} normais")
    
    # Salvar dados
    logger.info(f"Salvando dataset {dataset_name}...")
    
    # Arrays numpy
    np.save(output_dir / f'X_train_{dataset_name}.npy', X_train)
    np.save(output_dir / f'X_test_{dataset_name}.npy', X_test)
    np.save(output_dir / f'y_train_multi_{dataset_name}.npy', y_train_multi)
    np.save(output_dir / f'y_test_multi_{dataset_name}.npy', y_test_multi)
    np.save(output_dir / f'y_train_binary_{dataset_name}.npy', y_train_bin)
    np.save(output_dir / f'y_test_binary_{dataset_name}.npy', y_test_bin)
    
    # Metadados
    metadata = {
        'dataset_name': dataset_name,
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y_encoded)),
        'attack_ratio': float(y_binary.mean()),
        'normal_ratio': float((y_binary == 0).mean()),
        'classes': le.classes_.tolist(),
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
        'data_characteristics': 'realistic_with_noise_and_overlap'
    }
    
    with open(output_dir / f'metadata_{dataset_name}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Salvar scaler e label encoder
    with open(output_dir / f'scaler_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(output_dir / f'label_encoder_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    logger.info(f"Dataset {dataset_name} salvo:")
    logger.info(f"  Treino: {X_train.shape}")
    logger.info(f"  Teste: {X_test.shape}")
    logger.info(f"  Classes: {len(np.unique(y_encoded))}")
    logger.info(f"  Ratio ataques: {y_binary.mean():.2%}")
    
    return metadata

def main():
    """Função principal"""
    logger.info("Criando datasets REALISTAS e mais desafiadores...")
    
    # Diretório de saída
    output_dir = Path(__file__).parent.parent.parent / 'src' / 'datasets' / 'clean'
    
    try:
        # Criar dataset CIC-DDoS2019 realista
        X_cic, y_cic, types_cic = create_realistic_cicddos_sample()
        metadata_cic = preprocess_and_save(X_cic, y_cic, 'cicddos', output_dir)
        
        # Criar dataset NF-UNSW-NB15-v3 realista
        X_unsw, y_unsw, types_unsw = create_realistic_unsw_sample()
        metadata_unsw = preprocess_and_save(X_unsw, y_unsw, 'unsw', output_dir)
        
        # Criar dataset integrado BALANCEADO
        logger.info("Criando dataset integrado balanceado...")
        
        # Alinhar número de features
        min_features = min(X_cic.shape[1], X_unsw.shape[1])
        X_cic_aligned = X_cic[:, :min_features]
        X_unsw_aligned = X_unsw[:, :min_features]
        
        # Balancear tamanhos dos datasets antes de combinar
        min_samples = min(len(X_cic_aligned), len(X_unsw_aligned))
        indices_cic = np.random.choice(len(X_cic_aligned), min_samples, replace=False)
        indices_unsw = np.random.choice(len(X_unsw_aligned), min_samples, replace=False)
        
        X_integrated = np.vstack([
            X_cic_aligned[indices_cic],
            X_unsw_aligned[indices_unsw]
        ])
        y_integrated = np.concatenate([
            ['CIC_' + label for label in y_cic[indices_cic]],
            ['UNSW_' + label for label in y_unsw[indices_unsw]]
        ])
        
        metadata_integrated = preprocess_and_save(X_integrated, y_integrated, 'integrated', output_dir)
        
        # Criar arquivo de resumo
        summary = {
            'created_datasets': ['cicddos', 'unsw', 'integrated'],
            'characteristics': 'realistic_with_noise_overlap_and_balanced_classes',
            'cicddos': metadata_cic,
            'unsw': metadata_unsw,
            'integrated': metadata_integrated,
            'total_size_mb': sum([
                X_cic.nbytes, X_unsw.nbytes, X_integrated.nbytes
            ]) / (1024 * 1024),
            'improvements': [
                'Added realistic noise to all datasets',
                'Introduced overlap between normal and attack traffic',
                'Balanced class distributions (20-40% normal traffic)',
                'Clipped outliers to prevent extreme values',
                'Stratified train/test splits'
            ]
        }
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Datasets REALISTAS criados com sucesso!")
        logger.info(f"Total: {summary['total_size_mb']:.1f} MB")
        logger.info(f"Melhorias: Ruído, overlap, classes balanceadas")
        logger.info(f"Localização: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao criar datasets realistas: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
