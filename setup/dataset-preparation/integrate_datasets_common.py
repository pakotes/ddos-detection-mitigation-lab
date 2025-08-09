#!/usr/bin/env python3
"""
Integração Avançada de Datasets: Seleção de Features Comuns

Este script integra múltiplos datasets processados, selecionando apenas as features comuns (por nome) entre todos os conjuntos, para treino híbrido robusto e consistente.
"""

import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CommonFeatureIntegrator:
    def __init__(self, processed_dir=None, output_dir=None):
        if processed_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            self.processed_dir = base_dir / "src" / "datasets" / "processed"
        else:
            self.processed_dir = Path(processed_dir)
        if output_dir is None:
            self.output_dir = self.processed_dir / "integrated"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_metadata(self, dataset_name):
        metadata_file = self.processed_dir / f"metadata_{dataset_name}.json"
        if not metadata_file.exists():
            return None
        with open(metadata_file, 'r') as f:
            return json.load(f)

    def load_data(self, dataset_name):
        X_file = self.processed_dir / f"X_{dataset_name}.npy"
        y_file = self.processed_dir / f"y_{dataset_name}.npy"
        if not X_file.exists() or not y_file.exists():
            return None, None
        X = np.load(X_file)
        y = np.load(y_file)
        return X, y

    def get_common_features(self, dataset_names):
        feature_sets = []
        for name in dataset_names:
            meta = self.load_metadata(name)
            if meta is None:
                logger.warning(f"Metadados não encontrados para {name}")
                continue
            # Tenta encontrar nomes das features
            if 'nomes_features' in meta:
                feature_sets.append(set(meta['nomes_features']))
            elif 'colunas_features' in meta:
                feature_sets.append(set(meta['colunas_features']))
            else:
                logger.warning(f"Features não encontradas nos metadados de {name}")
        if not feature_sets:
            raise ValueError("Nenhum conjunto de features encontrado nos metadados.")
        # Interseção de todos os conjuntos
        common = set.intersection(*feature_sets)
        logger.info(f"Features comuns encontradas: {len(common)}")
        return sorted(list(common))

    def integrate(self, dataset_names):
        logger.info(f"A integrar datasets: {dataset_names}")
        common_features = self.get_common_features(dataset_names)
        if not common_features:
            raise ValueError("Não existem features comuns entre os datasets.")
        integrated_X = []
        integrated_y = []
        source_info = []
        for name in dataset_names:
            meta = self.load_metadata(name)
            X, y = self.load_data(name)
            if X is None or y is None or meta is None:
                logger.warning(f"Dados incompletos para {name}, ignorado.")
                continue
            # Obter índice das features comuns
            if 'nomes_features' in meta:
                feature_names = meta['nomes_features']
            elif 'colunas_features' in meta:
                feature_names = meta['colunas_features']
            else:
                logger.warning(f"Features não encontradas nos metadados de {name}")
                continue
            idxs = [feature_names.index(f) for f in common_features if f in feature_names]
            if not idxs:
                logger.warning(f"Nenhuma feature comum encontrada em {name}")
                continue
            X_common = X[:, idxs]
            integrated_X.append(X_common)
            integrated_y.append(y)
            source_info.append({
                'dataset': name,
                'samples': int(X_common.shape[0]),
                'features': len(common_features)
            })
        if not integrated_X:
            raise ValueError("Nenhum dado válido para integração.")
        X_final = np.vstack(integrated_X)
        y_final = np.hstack(integrated_y)
        # Shuffle
        idx = np.random.permutation(len(X_final))
        X_final = X_final[idx]
        y_final = y_final[idx]
        # Guardar
        np.save(self.output_dir / "X_integrated_common.npy", X_final)
        np.save(self.output_dir / "y_integrated_common.npy", y_final)
        with open(self.output_dir / "feature_names_integrated_common.txt", "w") as f:
            f.write('\n'.join(common_features))
        metadata = {
            'configuration': 'integrated_common_features',
            'description': 'Dataset integrado apenas com features comuns',
            'total_samples': int(len(y_final)),
            'total_features': len(common_features),
            'source_datasets': source_info
        }
        with open(self.output_dir / "metadata_integrated_common.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Integração concluída: {X_final.shape[0]:,} amostras, {X_final.shape[1]} features comuns")
        return metadata

if __name__ == "__main__":
    # Nomes dos datasets processados (ajuste conforme necessário)
    DATASETS = ["nf_unsw", "cic_ddos", "nf_bot_iot"]
    integrator = CommonFeatureIntegrator()
    metadata = integrator.integrate(DATASETS)
    print("Integração avançada concluída!")
    print(f"Amostras: {metadata['total_samples']:,}")
    print(f"Features comuns: {metadata['total_features']}")
    print(f"Datasets de origem: {[d['dataset'] for d in metadata['source_datasets']]}")
    print(f"Diretório de saída: {integrator.output_dir}")
