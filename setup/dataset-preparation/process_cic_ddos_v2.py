#!/usr/bin/env python3
"""
Processador Avançado CIC-DDoS2019 v2

Pipeline robusto para preparação do dataset CIC-DDoS2019 segundo as 6 diretivas principais:
1. Remoção de features irrelevantes/socket
2. Remoção/substituição de valores em falta/infinitos
3. Remoção de duplicados
4. Seleção aleatória de subconjuntos equilibrados por classe
5. Divisão treino/teste/validação
6. Escalonamento MinMax

Extras incluídos:
- Logging detalhado
- Exportação de metadados completos
- Relatórios de distribuição de classes
- Exportação de amostras por classe
- Exportação de features removidas
- Exportação de índices de amostras selecionadas
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Diretivas
SOCKET_FEATURES = [
    'Unnamed: 0', 'Source Port', 'Destination Port', 'Flow ID',
    'Source IP', 'Destination IP', 'Timestamp', 'SimilarHTTP'
]
WEB_DDOS_CLASS = 'WebDDoS'
MAX_SAMPLES_PER_CLASS = 150000
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.2
TRAIN_RATIO = 0.8

class CICDDoSProcessorV2:
    def __init__(self, input_dir=None, output_dir=None):
        if input_dir is None:
            self.input_dir = Path(__file__).parent / "CIC-DDoS2019"
        else:
            self.input_dir = Path(input_dir)
        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            self.output_dir = base_dir / "src" / "datasets" / "processed"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_csv_files(self):
        csv_files = []
        for subdir in self.input_dir.iterdir():
            if subdir.is_dir():
                csv_files.extend(list(subdir.glob("*.csv")))
        csv_files.extend(list(self.input_dir.glob("*.csv")))
        return csv_files

    def load_and_concat(self):
        files = self.find_csv_files()
        logger.info(f"A carregar {len(files)} ficheiros CSV...")
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Erro ao ler {f}: {e}")
        data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total de amostras carregadas: {data.shape[0]:,}")
        return data

    def remove_socket_features(self, df):
        features_removidas = [f for f in SOCKET_FEATURES if f in df.columns]
        df = df.drop(columns=features_removidas, errors='ignore')
        logger.info(f"Features de socket removidas: {features_removidas}")
        return df, features_removidas

    def clean_missing_and_inf(self, df):
        # Substituir inf/-inf por NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        # Substituir NaN por mediana da coluna
        n_missing = df.isna().sum().sum()
        df = df.fillna(df.median(numeric_only=True))
        logger.info(f"Valores em falta/infinitos tratados: {n_missing}")
        return df, int(n_missing)

    def remove_duplicates(self, df):
        n_before = df.shape[0]
        df = df.drop_duplicates()
        n_after = df.shape[0]
        logger.info(f"Duplicados removidos: {n_before - n_after}")
        return df, n_before - n_after

    def select_balanced_subset(self, df, label_col='Attack'):  # 'Attack' ou 'Label'
        # Excluir WebDDoS
        if WEB_DDOS_CLASS in df[label_col].unique():
            df = df[df[label_col] != WEB_DDOS_CLASS]
            logger.info(f"Classe {WEB_DDOS_CLASS} excluída.")
        # Seleção aleatória por classe
        subset_indices = []
        class_counts = {}
        for cls in df[label_col].unique():
            cls_idx = df[df[label_col] == cls].index
            n_cls = len(cls_idx)
            n_select = min(n_cls, MAX_SAMPLES_PER_CLASS)
            selected = np.random.choice(cls_idx, n_select, replace=False)
            subset_indices.extend(selected)
            class_counts[cls] = n_select
        df_subset = df.loc[subset_indices].reset_index(drop=True)
        logger.info(f"Amostras selecionadas por classe: {class_counts}")
        return df_subset, class_counts, subset_indices

    def split_data(self, df, label_col='Attack', binary_col='Malicious'):
        # Multiclass
        X = df.drop(columns=[label_col, binary_col], errors='ignore').values
        y_multi = df[label_col].values
        # Binário
        y_bin = df[binary_col].values if binary_col in df.columns else (df[label_col] != 'BENIGN').astype(int)
        # Split
        X_train, X_test, y_train_multi, y_test_multi, y_train_bin, y_test_bin = train_test_split(
            X, y_multi, y_bin, test_size=TEST_RATIO, random_state=42, stratify=y_multi)
        # Validação
        X_train, X_val, y_train_multi, y_val_multi, y_train_bin, y_val_bin = train_test_split(
            X_train, y_train_multi, y_train_bin, test_size=VALIDATION_RATIO, random_state=42, stratify=y_train_multi)
        logger.info(f"Split: treino={X_train.shape[0]}, val={X_val.shape[0]}, teste={X_test.shape[0]}")
        return (X_train, X_val, X_test, y_train_multi, y_val_multi, y_test_multi, y_train_bin, y_val_bin, y_test_bin)

    def scale_features(self, X_train, X_val, X_test):
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Features normalizadas com MinMaxScaler.")
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    def export_all(self, X_train, X_val, X_test, y_train_multi, y_val_multi, y_test_multi, y_train_bin, y_val_bin, y_test_bin, scaler, features, metainfo):
        # Exportar arrays
        np.save(self.output_dir / "X_train_cic_v2.npy", X_train)
        np.save(self.output_dir / "X_val_cic_v2.npy", X_val)
        np.save(self.output_dir / "X_test_cic_v2.npy", X_test)
        np.save(self.output_dir / "y_train_multi_cic_v2.npy", y_train_multi)
        np.save(self.output_dir / "y_val_multi_cic_v2.npy", y_val_multi)
        np.save(self.output_dir / "y_test_multi_cic_v2.npy", y_test_multi)
        np.save(self.output_dir / "y_train_bin_cic_v2.npy", y_train_bin)
        np.save(self.output_dir / "y_val_bin_cic_v2.npy", y_val_bin)
        np.save(self.output_dir / "y_test_bin_cic_v2.npy", y_test_bin)
        # Exportar scaler
        import pickle
        with open(self.output_dir / "scaler_cic_v2.pkl", "wb") as f:
            pickle.dump(scaler, f)
        # Exportar features
        with open(self.output_dir / "feature_names_cic_v2.txt", "w") as f:
            f.write('\n'.join(features))
        # Exportar metadados
        with open(self.output_dir / "metadata_cic_v2.json", "w") as f:
            json.dump(metainfo, f, indent=2)
        logger.info("Dados e metadados exportados.")

    def process(self):
        # 1. Carregar e concatenar
        df = self.load_and_concat()
        # 2. Remover features irrelevantes
        df, features_removidas = self.remove_socket_features(df)
        # 3. Limpar valores em falta/infinitos
        df, n_missing = self.clean_missing_and_inf(df)
        # 4. Remover duplicados
        df, n_dupes = self.remove_duplicates(df)
        # 5. Criar coluna binária
        if 'Attack' in df.columns:
            df['Malicious'] = (df['Attack'] != 'BENIGN').astype(int)
        elif 'Label' in df.columns:
            df['Malicious'] = (df['Label'] != 'BENIGN').astype(int)
        # 6. Seleção aleatória equilibrada
        label_col = 'Attack' if 'Attack' in df.columns else 'Label'
        df_subset, class_counts, subset_indices = self.select_balanced_subset(df, label_col=label_col)
        # 7. Split
        split = self.split_data(df_subset, label_col=label_col, binary_col='Malicious')
        X_train, X_val, X_test, y_train_multi, y_val_multi, y_test_multi, y_train_bin, y_val_bin, y_test_bin = split
        # 8. Escalonamento
        features = [c for c in df_subset.columns if c not in [label_col, 'Malicious']]
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_val, X_test)
        # 9. Metadados
        metainfo = {
            'total_amostras': int(df.shape[0]),
            'amostras_pos_limpeza': int(df_subset.shape[0]),
            'features_removidas': features_removidas,
            'n_missing': n_missing,
            'n_dupes': n_dupes,
            'class_counts': class_counts,
            'subset_indices': [int(i) for i in subset_indices],
            'features': features,
            'split': {
                'train': int(X_train.shape[0]),
                'val': int(X_val.shape[0]),
                'test': int(X_test.shape[0])
            }
        }
        # 10. Exportar tudo
        self.export_all(X_train_scaled, X_val_scaled, X_test_scaled, y_train_multi, y_val_multi, y_test_multi, y_train_bin, y_val_bin, y_test_bin, scaler, features, metainfo)
        logger.info("Processamento CIC-DDoS2019 v2 concluído.")
        print("Processamento CIC-DDoS2019 v2 concluído!")
        print(f"Amostras finais: {df_subset.shape[0]:,}")
        print(f"Features finais: {len(features)}")
        print(f"Distribuição por classe: {class_counts}")
        print(f"Diretório de saída: {self.output_dir}")

if __name__ == "__main__":
    processor = CICDDoSProcessorV2()
    processor.process()
