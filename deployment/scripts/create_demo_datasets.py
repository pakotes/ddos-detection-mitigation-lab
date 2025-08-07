#!/usr/bin/env python3
"""
Criar datasets de demonstração para quando os dados reais não estão disponíveis
Útil para desenvolvimento e testes locais
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import joblib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

class DemoDatasetGenerator:
    """Gerador de datasets de demonstração para DDoS"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent / 'src' / 'datasets'
        self.demo_dir = self.base_path / 'demo'
        
    def create_demo_dir(self):
        """Criar diretório demo"""
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Diretório demo: {self.demo_dir}")
    
    def generate_ddos_features(self, n_samples=10000, n_features=20):
        """Gerar features sintéticas que simulam tráfego DDoS"""
        logger.info(f"Gerando {n_samples} amostras com {n_features} features...")
        
        # Usar make_classification para base
        X_base, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=3,
            n_clusters_per_class=2,
            class_sep=0.8,
            random_state=42
        )
        
        # Tornar features mais realistas para tráfego de rede
        feature_names = [
            'packet_size_avg', 'packet_size_std', 'packet_rate',
            'duration', 'bytes_sent', 'bytes_recv', 
            'packets_sent', 'packets_recv', 'flow_duration',
            'inter_arrival_time_avg', 'inter_arrival_time_std',
            'tcp_flags_syn', 'tcp_flags_ack', 'tcp_flags_fin',
            'port_src_entropy', 'port_dst_entropy',
            'protocol_tcp', 'protocol_udp', 'protocol_icmp',
            'connection_rate'
        ]
        
        # Ajustar para ter exatamente o número de features desejado
        if len(feature_names) > n_features:
            feature_names = feature_names[:n_features]
        elif len(feature_names) < n_features:
            for i in range(len(feature_names), n_features):
                feature_names.append(f'feature_{i}')
        
        # Normalizar features para valores realistas
        X = np.abs(X_base)  # Garantir valores positivos
        
        # Ajustar algumas features para ter distribuições mais realistas
        for i in range(n_features):
            if 'size' in feature_names[i]:
                X[:, i] = X[:, i] * 1500 + 64  # Tamanhos de pacote típicos
            elif 'rate' in feature_names[i]:
                X[:, i] = X[:, i] * 1000  # Taxa de pacotes
            elif 'duration' in feature_names[i]:
                X[:, i] = X[:, i] * 300  # Duração em segundos
            elif 'entropy' in feature_names[i]:
                X[:, i] = (X[:, i] / X[:, i].max()) * 8  # Entropia 0-8
            elif 'protocol' in feature_names[i]:
                X[:, i] = (X[:, i] > X[:, i].mean()).astype(int)  # Binário
        
        logger.info(f"Features geradas: {feature_names}")
        logger.info(f"Distribuição de classes: Normal={np.sum(y==0)}, DDoS={np.sum(y==1)}")
        
        return X, y, feature_names
    
    def create_realistic_splits(self, X, y, test_size=0.2, val_size=0.1):
        """Criar splits treino/validação/teste"""
        # Primeiro split: treino+val vs teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Segundo split: treino vs validação
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Splits criados:")
        logger.info(f"  Treino: {X_train.shape[0]} amostras")
        logger.info(f"  Validação: {X_val.shape[0]} amostras")
        logger.info(f"  Teste: {X_test.shape[0]} amostras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_dataset(self, name, X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
        """Salvar dataset completo"""
        logger.info(f"Salvando dataset demo: {name}")
        
        # Salvar arrays numpy
        np.save(self.demo_dir / f'X_train_{name}.npy', X_train)
        np.save(self.demo_dir / f'X_val_{name}.npy', X_val)
        np.save(self.demo_dir / f'X_test_{name}.npy', X_test)
        np.save(self.demo_dir / f'y_train_{name}.npy', y_train)
        np.save(self.demo_dir / f'y_val_{name}.npy', y_val)
        np.save(self.demo_dir / f'y_test_{name}.npy', y_test)
        
        # Criar versão binária dos labels (compatibilidade)
        np.save(self.demo_dir / f'y_train_binary_{name}.npy', y_train)
        np.save(self.demo_dir / f'y_val_binary_{name}.npy', y_val)
        np.save(self.demo_dir / f'y_test_binary_{name}.npy', y_test)
        
        # Salvar nomes das features
        with open(self.demo_dir / f'feature_names_{name}.txt', 'w') as f:
            for fname in feature_names:
                f.write(f"{fname}\n")
        
        # Criar metadata
        metadata = {
            'dataset_name': name,
            'n_samples_train': len(X_train),
            'n_samples_val': len(X_val),
            'n_samples_test': len(X_test),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'class_distribution_train': {
                'normal': int(np.sum(y_train == 0)),
                'ddos': int(np.sum(y_train == 1))
            },
            'dataset_type': 'demo',
            'generated_date': pd.Timestamp.now().isoformat()
        }
        
        with open(self.demo_dir / f'metadata_{name}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_integrated_version(self):
        """Criar versão integrada dos dados demo"""
        logger.info("Criando versão integrada dos dados demo...")
        
        integrated_dir = self.base_path / 'integrated'
        integrated_dir.mkdir(parents=True, exist_ok=True)
        
        # Combinar dados de demo se existirem
        all_X = []
        all_y = []
        feature_names = None
        
        for name in ['cicddos', 'unsw']:
            X_file = self.demo_dir / f'X_train_{name}.npy'
            if X_file.exists():
                X = np.load(X_file)
                y = np.load(self.demo_dir / f'y_train_{name}.npy')
                
                # Carregar feature names
                feature_file = self.demo_dir / f'feature_names_{name}.txt'
                with open(feature_file, 'r') as f:
                    current_features = [line.strip() for line in f.readlines()]
                
                if feature_names is None:
                    feature_names = current_features
                
                all_X.append(X)
                all_y.append(y)
        
        if not all_X:
            logger.warning("Nenhum dado demo encontrado para integração")
            return
        
        # Combinar dados
        X_integrated = np.vstack(all_X)
        y_integrated = np.concatenate(all_y)
        
        # Salvar versão integrada
        np.save(integrated_dir / 'X_integrated_demo.npy', X_integrated)
        np.save(integrated_dir / 'y_integrated_demo.npy', y_integrated)
        
        # Salvar feature names
        with open(integrated_dir / 'feature_names_demo.txt', 'w') as f:
            for fname in feature_names:
                f.write(f"{fname}\n")
        
        # Metadata integrada
        metadata = {
            'dataset_type': 'integrated_demo',
            'n_samples': len(X_integrated),
            'n_features': X_integrated.shape[1],
            'feature_names': feature_names,
            'class_distribution': {
                'normal': int(np.sum(y_integrated == 0)),
                'ddos': int(np.sum(y_integrated == 1))
            },
            'generated_date': pd.Timestamp.now().isoformat()
        }
        
        with open(integrated_dir / 'metadata_demo.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dados integrados salvos: {X_integrated.shape}")
    
    def generate_all_demo_datasets(self):
        """Gerar todos os datasets de demonstração"""
        logger.info("Gerando datasets de demonstração completos...")
        
        self.create_demo_dir()
        
        # Dataset tipo CIC-DDoS (mais features de rede)
        logger.info("\nGerando dataset demo CIC-DDoS...")
        X, y, features = self.generate_ddos_features(n_samples=15000, n_features=25)
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_realistic_splits(X, y)
        self.save_dataset('cicddos', X_train, X_val, X_test, y_train, y_val, y_test, features)
        
        # Dataset tipo UNSW (features mais diversas)
        logger.info("\nGerando dataset demo UNSW...")
        X, y, features = self.generate_ddos_features(n_samples=12000, n_features=20)
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_realistic_splits(X, y)
        self.save_dataset('unsw', X_train, X_val, X_test, y_train, y_val, y_test, features)
        
        # Criar versão integrada
        self.create_integrated_version()
        
        # Criar summary
        summary = {
            'demo_datasets_created': ['cicddos', 'unsw'],
            'integrated_version': 'available',
            'purpose': 'demonstration and local development',
            'note': 'These are synthetic datasets for demo purposes only',
            'created_date': pd.Timestamp.now().isoformat()
        }
        
        with open(self.demo_dir / 'summary_demo.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("\nDATASETS DE DEMONSTRAÇÃO CRIADOS!")
        logger.info(f"📁 Localização: {self.demo_dir}")
        logger.info("Versão integrada disponível em: src/datasets/integrated/")
        logger.info("NOTA: Estes são dados sintéticos apenas para demonstração")

def main():
    """Função principal"""
    logger.info("CRIADOR DE DATASETS DE DEMONSTRAÇÃO")
    logger.info("="*50)
    
    try:
        generator = DemoDatasetGenerator()
        generator.generate_all_demo_datasets()
        
        logger.info("\nPRÓXIMOS PASSOS:")
        logger.info("1. Execute: python deployment/scripts/create_production_model.py")
        logger.info("2. Teste: python deployment/scripts/test_production_model.py")
        logger.info("3. Pipeline completo: python deployment/scripts/production_pipeline.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro gerando dados demo: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
