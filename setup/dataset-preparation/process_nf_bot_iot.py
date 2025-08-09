#!/usr/bin/env python3
"""
Processador NF-BoT-IoT
Processa o CSV em batches, exporta arrays .npy e metadados, usando ficheiro de features.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Função leve para otimizar tipos de dados de DataFrames
def df_shrink(df, obj2cat=False, int2uint=False):
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['integer']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    if obj2cat:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
    if int2uint:
        for col in df.select_dtypes(include=['int']).columns:
            if (df[col] >= 0).all():
                df[col] = pd.to_numeric(df[col], downcast='unsigned')
    return df

class NFBoTIoTProcessor:
    def __init__(self, input_dir=None, output_dir=None, chunksize=500000):
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
        logger.info("A processar NF-BoT-IoT em batches...")
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
        # Carregar todo o CSV para limpeza e amostragem estratificada
        df = pd.read_csv(csv_file)
        # Identificar colunas categóricas
        categorical_cols = [
            'proto', 'flgs', 'state', 'smac', 'dmac', 'saddr', 'daddr', 'category', 'subcategory', 'stime', 'ltime'
        ]
        # Limpeza centralizada
        drop_cols = [col for col in df.columns if col.lower() in ['id', 'index', 'unnamed: 0']]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        df = df_shrink(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)
        # Amostragem estratificada
        n_amostra = 2_500_000
        if 'Label' in df.columns:
            # Proporção entre classes
            df_sample = df.groupby('Label', group_keys=False).apply(lambda x: x.sample(frac=min(1, n_amostra/len(df)), random_state=42))
            logger.info(f"Amostragem estratificada: {len(df_sample)} registos selecionados")
            df = df_sample.reset_index(drop=True)
        # Processar batches
        batch_idx = 1
        batch_files_X = []
        batch_files_y = []
        n_samples = 0
        attack_count = 0
        attack_names = set()
        # Selecionar apenas colunas numéricas
        feature_cols = [col for col in df.columns if col not in categorical_cols + ['Label', 'Attack']]
        numeric_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
        # Processar em batches
        for start in range(0, len(df), self.chunksize):
            end = min(start + self.chunksize, len(df))
            chunk = df.iloc[start:end]
            X_batch = chunk[numeric_cols].values.astype(np.float32)
            y_batch = chunk['Label'].values.astype(np.float32) if 'Label' in chunk.columns else np.zeros(len(chunk), dtype=np.float32)
            if 'Attack' in chunk.columns:
                attack_names.update([a for a in chunk['Attack'].unique() if a != 'Benign'])
            n_samples += X_batch.shape[0]
            attack_count += int(np.sum(y_batch))
            logger.info(f"Batch {batch_idx} processado: {X_batch.shape[0]} amostras")
            batch_X_file = self.output_dir / f"X_nf_bot_iot_batch_{batch_idx}.npy"
            batch_y_file = self.output_dir / f"y_nf_bot_iot_batch_{batch_idx}.npy"
            np.save(batch_X_file, X_batch)
            np.save(batch_y_file, y_batch)
            batch_files_X.append(str(batch_X_file))
            batch_files_y.append(str(batch_y_file))
            batch_idx += 1
        # Exportar nomes dos ataques
        with open(self.output_dir / "attack_names_nf_bot_iot.txt", "w") as f:
            f.write('\n'.join(sorted(attack_names)))
        # Exportar lista de ficheiros de batches
        with open(self.output_dir / "X_nf_bot_iot_batches.txt", "w") as f:
            f.write('\n'.join(batch_files_X))
        with open(self.output_dir / "y_nf_bot_iot_batches.txt", "w") as f:
            f.write('\n'.join(batch_files_y))

        # União sequencial dos batches para evitar consumo excessivo de memória
        logger.info("A unir todos os batches em arrays únicos de forma sequencial...")
        import os
        import numpy.lib.format
        # Determinar shapes
        total_samples = 0
        X_shape = None
        for x_file in batch_files_X:
            arr = np.load(x_file)
            if X_shape is None:
                X_shape = arr.shape[1]
            total_samples += arr.shape[0]
        # Criar ficheiros .npy em modo append
        X_out_path = self.output_dir / "X_nf_bot_iot.npy"
        y_out_path = self.output_dir / "y_nf_bot_iot.npy"
        # Criar ficheiro X
        X_out = np.lib.format.open_memmap(X_out_path, mode='w+', dtype='float32', shape=(total_samples, X_shape))
        y_out = np.lib.format.open_memmap(y_out_path, mode='w+', dtype='float32', shape=(total_samples,))
        idx = 0
        for x_file, y_file in zip(batch_files_X, batch_files_y):
            X_batch = np.load(x_file)
            y_batch = np.load(y_file)
            X_out[idx:idx+X_batch.shape[0], :] = X_batch
            y_out[idx:idx+X_batch.shape[0]] = y_batch
            idx += X_batch.shape[0]
        logger.info(f"Arrays únicos exportados: X_nf_bot_iot.npy ({X_out.shape}), y_nf_bot_iot.npy ({y_out.shape})")
        # Eliminar ficheiros batch após sucesso
        for f in batch_files_X + batch_files_y:
            try:
                os.remove(f)
            except Exception as e:
                logger.warning(f"Não foi possível eliminar {f}: {e}")
        logger.info("Ficheiros batch eliminados após união.")
        # Exportar nomes das features
        if feature_names:
            with open(self.output_dir / "feature_names_nf_bot_iot.txt", "w") as f:
                f.write('\n'.join(feature_names))
        else:
            with open(self.output_dir / "feature_names_nf_bot_iot.txt", "w") as f:
                f.write('\n'.join(feature_cols))
        n_features = len(feature_cols)
        normal_count = int(n_samples - attack_count)
        attack_ratio = float(attack_count / n_samples) if n_samples > 0 else 0.0
        # Metadados completos
        metadata = {
            "dataset": "nf_bot_iot",
            "colunas": list(chunk.columns),
            "colunas_features": feature_cols,
            "colunas_categoricas": [col for col in categorical_cols if col in chunk.columns],
            "alvo": "Label",
            "feature_names_file": "feature_names_nf_bot_iot.txt",
            "attack_names_file": "attack_names_nf_bot_iot.txt",
            "batch_files_X": batch_files_X,
            "batch_files_y": batch_files_y,
            "amostras": n_samples,
            "features": n_features,
            "amostras_ataque": attack_count,
            "amostras_normais": normal_count,
            "percentagem_ataque": attack_ratio
        }
        with open(self.output_dir / "metadata_nf_bot_iot.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Processamento de NF-BoT-IoT concluído com sucesso. Amostras: {n_samples}, Features: {n_features}, Ataques: {attack_count}, Normais: {normal_count}, Percentagem ataque: {attack_ratio:.2%}")
        print(f"Processamento de NF-BoT-IoT concluído com sucesso. Amostras: {n_samples}, Features: {n_features}, Ataques: {attack_count}, Normais: {normal_count}, Percentagem ataque: {attack_ratio:.2%}")
        return True
    # (Removido: bloco de amostragem estratificada fora do método)

if __name__ == "__main__":
    print("[DEBUG] Início do processamento NF-BoT-IoT")
    processor = NFBoTIoTProcessor()
    print(f"[DEBUG] Diretório de entrada: {processor.input_dir}")
    print(f"[DEBUG] Diretório de saída: {processor.output_dir}")
    resultado = processor.process()
    print(f"[DEBUG] Resultado do processamento: {resultado}")
