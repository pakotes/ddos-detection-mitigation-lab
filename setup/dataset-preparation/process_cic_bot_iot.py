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
            self.input_dir = Path(__file__).parent / "BoT-IoT"
        else:
            self.input_dir = Path(input_dir)
        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            self.output_dir = base_dir / "src" / "datasets" / "processed"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_csv_files(self):
        """Procura todos os ficheiros CSV no diretório de input."""
        csv_files = list(self.input_dir.glob("*.csv"))
        return csv_files

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
            logger.info(f"A carregar {csv_file.name}")
            df = pd.read_csv(csv_file, low_memory=False)
            # Excluir dataframes vazios ou só com NA
            if not df.empty and not df.isna().all().all():
                dfs.append(df)
            else:
                logger.warning(f"Ficheiro ignorado (vazio ou só com NA): {csv_file.name}")
        if not dfs:
            logger.error("Todos os ficheiros CSV estão vazios ou só com NA.")
            return False
        data = pd.concat(dfs, ignore_index=True)

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
        logger.info(f"Processamento de CIC-BoT-IoT concluído com sucesso. Amostras: {n_samples}, Features: {n_features}, Ataques: {attack_count}, Normais: {normal_count}, Percentagem ataque: {attack_ratio:.2%}")
        return True

if __name__ == "__main__":
    processor = BoTIoTProcessor()
    success = processor.process()
    exit(0 if success else 1)
