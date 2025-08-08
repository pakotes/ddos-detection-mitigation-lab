#!/usr/bin/env python3
"""
Processador do Conjunto de Dados NF-UNSW-NB15-v3

Este módulo processa o conjunto de dados NetFlow NF-UNSW-NB15-v3 para
detecção geral de intrusões. Trata do carregamento de dados, pré-processamento,
engenharia de características e guarda os dados processados para uso em
modelos de aprendizagem automática.

"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NFUNSWProcessor:
    """
    Processador do conjunto de dados NF-UNSW-NB15-v3 para deteção de intrusões.
    Realiza o carregamento, limpeza, codificação e exportação dos dados prontos para treino de modelos.
    """
    
    def __init__(self, input_dir=None, output_dir=None):
        if input_dir is None:
            # Usar diretório local do dataset
            self.input_dir = Path(__file__).parent / "NF-UNSW-NB15-v3"
        else:
            self.input_dir = Path(input_dir)

        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            self.output_dir = base_dir / "src" / "datasets" / "processed"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_and_preprocess(self):
        """Carrega e pré-processa os dados NF-UNSW-NB15-v3"""
        logger.info("Início do processamento do NF-UNSW-NB15-v3...")

        # Procurar ficheiros CSV
        csv_files = list(self.input_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Não foram encontrados ficheiros CSV em {self.input_dir}")

        logger.info(f"Foram encontrados {len(csv_files)} ficheiros CSV")

        dataframes = []
        for csv_file in csv_files:
            logger.info(f"A carregar {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                if not df.empty:
                    dataframes.append(df)
                    logger.info(f"   Carregadas {df.shape[0]:,} linhas, {df.shape[1]} colunas")
            except Exception as e:
                logger.warning(f"   Erro ao carregar {csv_file.name}: {e}")

        if not dataframes:
            raise ValueError("Nenhum ficheiro válido foi carregado")

        # Combinar datasets
        logger.info("A combinar datasets...")
        df_combined = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Dataset combinado: {df_combined.shape[0]:,} linhas, {df_combined.shape[1]} colunas")

        return self._preprocess_features(df_combined)
    
    def _preprocess_features(self, df):
        """Pré-processa as features do dataset"""
        logger.info("Pré-processamento das features...")

        # Identificar coluna de label
        label_candidates = ['Label', ' Label', 'label', 'attack', 'Attack']
        label_col = None
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                break

        if label_col is None:
            raise ValueError("Coluna de label não encontrada")

        logger.info(f"A usar coluna de label: {label_col}")

        # Separar features e labels
        feature_cols = [col for col in df.columns if col != label_col]

        # Processar labels
        y_raw = df[label_col]
        if y_raw.dtype == 'object':
            # Labels categóricos
            normal_labels = ['BENIGN', 'Normal', 'normal', 'NORMAL']
            y_binary = (~y_raw.isin(normal_labels)).astype(int)
            logger.info("Convertidos labels categóricos para binário")
        else:
            # Labels numéricos
            y_binary = (y_raw != 0).astype(int)
            logger.info("Labels numéricos processados")

        # Processar features
        X = df[feature_cols].copy()

        # Selecionar apenas colunas numéricas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].copy()

        # Codificar colunas categóricas simples
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            logger.info(f"A codificar {len(categorical_cols)} colunas categóricas...")
            for col in categorical_cols:
                unique_count = X[col].nunique()
                if unique_count < 50:  # Limite razoável
                    try:
                        le = LabelEncoder()
                        X_numeric[f"{col}_codificada"] = le.fit_transform(X[col].fillna('desconhecido'))
                        logger.info(f"   Codificada {col} ({unique_count} valores)")
                    except:
                        logger.warning(f"   Erro ao codificar {col}")

        # Limpeza de dados
        logger.info("A limpar dados...")

        # Tratar infinitos
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

        # Tratar valores em falta
        missing_count = X_numeric.isnull().sum().sum()
        if missing_count > 0:
            X_numeric = X_numeric.fillna(X_numeric.median())
            logger.info(f"   Preenchidos {missing_count} valores em falta")

        # Remover features de variância zero
        variance = X_numeric.var()
        zero_var_cols = variance[variance == 0].index
        if len(zero_var_cols) > 0:
            X_numeric = X_numeric.drop(columns=zero_var_cols)
            logger.info(f"   Removidas {len(zero_var_cols)} features de variância zero")

        # Estatísticas finais
        attack_count = y_binary.sum()
        normal_count = len(y_binary) - attack_count
        attack_ratio = attack_count / len(y_binary)

        logger.info(f"Dataset final: {X_numeric.shape[0]:,} amostras, {X_numeric.shape[1]} features")
        logger.info(f"Normais: {normal_count:,} ({1-attack_ratio:.1%})")
        logger.info(f"Ataques: {attack_count:,} ({attack_ratio:.1%})")

        return X_numeric.values, y_binary.values, list(X_numeric.columns)
    
    def save_processed_data(self, X, y, feature_names):
        """Guarda os dados processados em disco"""
        logger.info("A guardar dados processados...")

        # Guardar arrays processados
        np.save(self.output_dir / "X_nf_unsw.npy", X)
        np.save(self.output_dir / "y_nf_unsw.npy", y)

        # Normalizar features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Guardar dados normalizados
        np.save(self.output_dir / "X_nf_unsw_scaled.npy", X_scaled)

        # Guardar scaler
        with open(self.output_dir / "scaler_nf_unsw.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        # Guardar nomes das features
        with open(self.output_dir / "feature_names_nf_unsw.txt", 'w') as f:
            f.write('\n'.join(feature_names))

        # Guardar metadados
        metadata = {
            'dataset': 'NF-UNSW-NB15-v3',
            'amostras': int(X.shape[0]),
            'features': int(X.shape[1]),
            'amostras_ataque': int(y.sum()),
            'amostras_normais': int(len(y) - y.sum()),
            'percentagem_ataque': float(y.sum() / len(y)),
            'nomes_features': feature_names
        }

        with open(self.output_dir / "metadata_nf_unsw.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dados guardados em {self.output_dir}")
        logger.info(f"Total de amostras: {X.shape[0]:,}")
        logger.info(f"Features: {X.shape[1]}")

        return metadata

def main():
    """Executa o pipeline de processamento do NF-UNSW-NB15-v3"""
    try:
        processor = NFUNSWProcessor()

        # Verificar se o diretório de input existe
        if not processor.input_dir.exists():
            logger.error(f"Diretório de input não encontrado: {processor.input_dir}")
            logger.info("Por favor, certifique-se que o dataset NF-UNSW-NB15-v3 está disponível")
            return False

        # Processar dados
        X, y, feature_names = processor.load_and_preprocess()

        # Guardar dados processados
        metadata = processor.save_processed_data(X, y, feature_names)

        logger.info("Processamento do NF-UNSW-NB15-v3 concluído com sucesso")
        return True

    except Exception as e:
        logger.error(f"Falha no processamento: {e}")
        return False

if __name__ == "__main__":
    main()
