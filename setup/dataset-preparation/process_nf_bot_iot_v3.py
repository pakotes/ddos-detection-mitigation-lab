#!/usr/bin/env python3
"""
Processador NF-BoT-IoT-v3
Processa o CSV em batches, exporta arrays .npy e metadados, usando ficheiro de features.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NFBoTIoTProcessor:
    def __init__(self, input_dir=None, output_dir=None, chunksize=50000):
        if input_dir is None:
            self.input_dir = Path(__file__).parent / "NF-BoT-IoT-v3"
        else:
            self.input_dir = Path(input_dir)
        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            self.output_dir = base_dir / "src" / "datasets" / "processed"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunksize = chunksize

    def find_data_files(self):
        csv_file = next(self.input_dir.glob("*.csv"), None)
        features_csv = next((f for f in self.input_dir.glob("*.csv") if "feature" in f.name.lower()), None)
        return csv_file, features_csv

    def process(self):
        logger.info("A processar NF-BoT-IoT-v3 em batches...")
        csv_file, features_csv = self.find_data_files()
        if not csv_file:
            logger.error(f"Nenhum ficheiro CSV encontrado em {self.input_dir}")
            print(f"ERRO: Nenhum ficheiro CSV encontrado em {self.input_dir}")
            return False
        logger.info(f"A carregar {csv_file.name} em batches de {self.chunksize}")
        # Carregar nomes das features do CSV se existir
        feature_names = None
        if features_csv:
            logger.info(f"A carregar nomes das features de {features_csv.name}")
            try:
                features_df = pd.read_csv(features_csv)
                for col in ['Feature', 'feature', 'name', 'Nome', 'nome']:
                    if col in features_df.columns:
                        feature_names = features_df[col].tolist()
                        break
            except Exception as e:
                logger.warning(f"Erro ao carregar CSV de features: {e}")
        # Processar batches
        batch_idx = 1
        batch_files_X = []
        batch_files_y = []
        n_samples = 0
        attack_count = 0
        for chunk in pd.read_csv(csv_file, chunksize=self.chunksize):
            # Identificar colunas categóricas
            categorical_cols = [
                'proto', 'flgs', 'state', 'smac', 'dmac', 'saddr', 'daddr', 'category', 'subcategory', 'stime', 'ltime'
            ]
            feature_cols = [col for col in chunk.columns if col not in categorical_cols + ['attack']]
            X_batch = chunk[feature_cols].values
            y_batch = chunk['attack'].values if 'attack' in chunk.columns else np.zeros(len(chunk))
            n_samples += X_batch.shape[0]
            attack_count += int(np.sum(y_batch))
            logger.info(f"Batch {batch_idx} processado: {X_batch.shape[0]} amostras")
            # Exportar batch
            batch_X_file = self.output_dir / f"X_nf_bot_iot_v3_batch_{batch_idx}.npy"
            batch_y_file = self.output_dir / f"y_nf_bot_iot_v3_batch_{batch_idx}.npy"
            np.save(batch_X_file, X_batch)
            np.save(batch_y_file, y_batch)
            batch_files_X.append(str(batch_X_file))
            batch_files_y.append(str(batch_y_file))
            batch_idx += 1
        # Exportar lista de ficheiros de batches
        with open(self.output_dir / "X_nf_bot_iot_v3_batches.txt", "w") as f:
            f.write('\n'.join(batch_files_X))
        with open(self.output_dir / "y_nf_bot_iot_v3_batches.txt", "w") as f:
            f.write('\n'.join(batch_files_y))
        # Exportar nomes das features
        if feature_names:
            with open(self.output_dir / "feature_names_nf_bot_iot_v3.txt", "w") as f:
                f.write('\n'.join(feature_names))
        else:
            with open(self.output_dir / "feature_names_nf_bot_iot_v3.txt", "w") as f:
                f.write('\n'.join(feature_cols))
        n_features = len(feature_cols)
        normal_count = int(n_samples - attack_count)
        attack_ratio = float(attack_count / n_samples) if n_samples > 0 else 0.0
        # Metadados completos
        metadata = {
            "dataset": "nf_bot_iot_v3",
            "colunas": list(chunk.columns),
            "colunas_features": feature_cols,
            "colunas_categoricas": [col for col in categorical_cols if col in chunk.columns],
            "alvo": "attack",
            "feature_names_file": "feature_names_nf_bot_iot_v3.txt",
            "batch_files_X": batch_files_X,
            "batch_files_y": batch_files_y,
            "amostras": n_samples,
            "features": n_features,
            "amostras_ataque": attack_count,
            "amostras_normais": normal_count,
            "percentagem_ataque": attack_ratio
        }
        with open(self.output_dir / "metadata_nf_bot_iot_v3.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Processamento de NF-BoT-IoT-v3 concluído com sucesso. Amostras: {n_samples}, Features: {n_features}, Ataques: {attack_count}, Normais: {normal_count}, Percentagem ataque: {attack_ratio:.2%}")
        print(f"Processamento de NF-BoT-IoT-v3 concluído com sucesso. Amostras: {n_samples}, Features: {n_features}, Ataques: {attack_count}, Normais: {normal_count}, Percentagem ataque: {attack_ratio:.2%}")
        return True

if __name__ == "__main__":
    print("[DEBUG] Início do processamento NF-BoT-IoT-v3")
    processor = NFBoTIoTProcessor()
    print(f"[DEBUG] Diretório de entrada: {processor.input_dir}")
    print(f"[DEBUG] Diretório de saída: {processor.output_dir}")
    resultado = processor.process()
    print(f"[DEBUG] Resultado do processamento: {resultado}")
