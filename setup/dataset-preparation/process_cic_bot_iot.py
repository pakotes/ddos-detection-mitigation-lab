#!/usr/bin/env python3
"""
Processador do Conjunto de Dados BoT-IoT

Este módulo processa o conjunto de dados BoT-IoT para deteção de ataques IoT e DDoS.
Inclui carregamento, pré-processamento, engenharia de características e exportação em formato compatível com o pipeline do laboratório.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BoTIoTProcessor:
    def __init__(self, input_dir=None, output_dir=None):
        if input_dir is None:
            self.input_dir = Path(__file__).parent / "CIC-BoT-IoT"
        else:
            self.input_dir = Path(input_dir)
        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            self.output_dir = base_dir / "src" / "datasets" / "processed"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_data_files(self):
        """Procura ficheiro Parquet principal e CSV de features."""
        parquet_files = list(self.input_dir.glob("*.parquet"))
        csv_files = list(self.input_dir.glob("*.csv"))
        parquet_file = parquet_files[0] if parquet_files else None
        features_csv = None
        for f in csv_files:
            if "feature" in f.name.lower():
                features_csv = f
                break
        return parquet_file, features_csv

    def process(self):
        """Processa o conjunto de dados BoT-IoT e guarda os resultados em disco."""
        logger.info("A processar BoT-IoT...")
        csv_files = self.find_csv_files()
        if not csv_files:
            logger.error("Nenhum ficheiro CSV encontrado em %s", self.input_dir)
            return False
        # Carregar e concatenar todos os ficheiros, excluindo dataframes vazios ou só com NA
        dfs = []
        for csv_file in csv_files:
            logger.info("A processar CIC-BoT-IoT...")
            parquet_file, features_csv = self.find_data_files()
            if not parquet_file:
                logger.error("Nenhum ficheiro Parquet encontrado em %s", self.input_dir)
                return False
            logger.info(f"A carregar {parquet_file.name}")
            data = pd.read_parquet(parquet_file)
            # Carregar nomes das features do CSV se existir
            feature_names = None
            if features_csv:
                logger.info(f"A carregar nomes das features de {features_csv.name}")
                try:
                    features_df = pd.read_csv(features_csv)
                    # Tentar encontrar coluna de nomes
                    for col in ['Feature', 'feature', 'name', 'Nome', 'nome']:
                        if col in features_df.columns:
                            feature_names = features_df[col].tolist()
                            break
                except Exception as e:
                    logger.warning(f"Erro ao carregar CSV de features: {e}")
            # Identificar colunas categóricas
            categorical_cols = [
                'proto', 'flgs', 'state', 'smac', 'dmac', 'saddr', 'daddr', 'category', 'subcategory', 'stime', 'ltime'
            ]
            # Excluir colunas categóricas e o target do X
            feature_cols = [col for col in data.columns if col not in categorical_cols + ['attack']]
            X = data[feature_cols].values
            y = data['attack'].values if 'attack' in data.columns else np.zeros(len(data))
            # Exportação compatível com pipeline
            np.save(self.output_dir / "X_cic_bot_iot.npy", X)
            np.save(self.output_dir / "y_cic_bot_iot.npy", y)
            # Exportar nomes das features
            if feature_names:
                with open(self.output_dir / "feature_names_cic_bot_iot.txt", "w") as f:
                    f.write('\n'.join(feature_names))
            else:
                with open(self.output_dir / "feature_names_cic_bot_iot.txt", "w") as f:
                    f.write('\n'.join(feature_cols))
            # Estatísticas básicas
            n_samples = X.shape[0]
            n_features = X.shape[1]
            attack_count = int(np.sum(y))
            normal_count = int(n_samples - attack_count)
            attack_ratio = float(attack_count / n_samples) if n_samples > 0 else 0.0
            # Metadados completos
            metadata = {
                "dataset": "CIC-BoT-IoT",
                "colunas": list(data.columns),
                "colunas_features": feature_cols,
                "colunas_categoricas": [col for col in categorical_cols if col in data.columns],
                "alvo": "attack",
                "dimensao": X.shape,
                "feature_names_file": "feature_names_cic_bot_iot.txt",
                "array_file": "X_cic_bot_iot.npy",
                "label_file": "y_cic_bot_iot.npy",
                "amostras": n_samples,
                "features": n_features,
                "amostras_ataque": attack_count,
                "amostras_normais": normal_count,
                "percentagem_ataque": attack_ratio
            }
            with open(self.output_dir / "metadata_cic_bot_iot.json", "w") as f:
                json.dump(metadata, f, indent=2)
                # Validação dos ficheiros exportados
                missing_files = []
                for fname in ["X_cic_bot_iot.npy", "y_cic_bot_iot.npy"]:
                    if not (self.output_dir / fname).exists():
                        missing_files.append(fname)
                if missing_files:
                    logger.error(f"Ficheiros de saída esperados em falta: {missing_files}")
                    print(f"ERRO: Ficheiros de saída esperados em falta: {missing_files}")
                    return False
            logger.info(f"Processamento de CIC-BoT-IoT concluído com sucesso. Amostras: {n_samples}, Features: {n_features}, Ataques: {attack_count}, Normais: {normal_count}, Percentagem ataque: {attack_ratio:.2%}")
            return True
