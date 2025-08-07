#!/usr/bin/env python3
"""
Processador do Conjunto de Dados CIC-DDoS2019

Este módulo processa o conjunto de dados CIC-DDoS2019 para detecção especializada
de DDoS. Trata do carregamento de dados, pré-processamento, engenharia de 
características e processamento em lotes para manuseamento eficiente de memória
de grandes conjuntos de dados.

O processador trata especificamente da inclusão de dados de tráfego BENIGN
que é essencial para treino realista de modelos de detecção de DDoS.

"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import gc

# Diagnóstico automático de headers dos CSV do CIC-DDoS2019
def diagnosticar_headers_csv(input_dir=None):
    """
    Diagnostica headers dos ficheiros CSV no diretório CIC-DDoS2019.
    Lista headers únicos e associa cada ficheiro ao respetivo header.
    Sinaliza ficheiros com headers diferentes.
    """
    from collections import defaultdict
    import pandas as pd
    from pathlib import Path

    if input_dir is None:
        input_dir = Path(__file__).parent / "CIC-DDoS2019"
    else:
        input_dir = Path(input_dir)

    # Encontrar todos os CSV
    csv_files = []
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            csv_files.extend(list(subdir.glob("*.csv")))
    csv_files.extend(list(input_dir.glob("*.csv")))

    if not csv_files:
        print(f"[Diagnóstico] Não foram encontrados ficheiros CSV em {input_dir}")
        return

    headers_dict = defaultdict(list)
    for f in csv_files:
        try:
            df = pd.read_csv(f, nrows=1, low_memory=False)
            header_tuple = tuple([c.strip() for c in df.columns])
            headers_dict[header_tuple].append(str(f))
        except Exception as e:
            print(f"[Diagnóstico] Erro ao ler {f}: {e}")

    print("\n[Diagnóstico] Headers únicos encontrados nos CSV:")
    for idx, (header, files) in enumerate(headers_dict.items(), 1):
        print(f"\nHeader #{idx} (ocorrências: {len(files)}):")
        print(f"  Colunas: {header}")
        print("  Ficheiros:")
        for file in files:
            print(f"    - {file}")

    if len(headers_dict) > 1:
        print("\n[Diagnóstico] AVISO: Foram encontrados múltiplos headers diferentes! Recomenda-se corrigir/remover os ficheiros problemáticos antes de processar o dataset.")
    else:
        print("\n[Diagnóstico] Todos os ficheiros CSV têm o mesmo header. Pronto para processamento.")

    print("\n[Diagnóstico] Fim do diagnóstico de headers.\n")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CICDDoSProcessor:
    """
    Processa o conjunto de dados CIC-DDoS2019 para detecção de ataques DDoS.
    
    Este processador é concebido para tratar o conjunto completo CIC-DDoS2019
    incluindo tanto amostras de ataque como amostras cruciais de tráfego BENIGN
    necessárias para classificação binária adequada.
    """
    
    def __init__(self, input_dir=None, output_dir=None, batch_size=10000):
        """
        Initialize the CIC-DDoS processor.
        
        Args:
            input_dir: Path to CIC-DDoS2019 dataset directory
            output_dir: Path to save processed data
            batch_size: Number of samples to process in each batch
        """
        if input_dir is None:
            # Use local dataset directory
            self.input_dir = Path(__file__).parent / "CIC-DDoS2019"
        else:
            self.input_dir = Path(input_dir)
            
        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            self.output_dir = base_dir / "src" / "datasets" / "processed"
        else:
            self.output_dir = Path(output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
    
    def find_csv_files(self):
        """Locate all CSV files in the dataset directory structure"""
        csv_files = []
        
        # Search in subdirectories (01-12, 03-11, etc.)
        for subdir in self.input_dir.iterdir():
            if subdir.is_dir():
                csv_files.extend(list(subdir.glob("*.csv")))
        
        # Also check root directory
        csv_files.extend(list(self.input_dir.glob("*.csv")))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.input_dir}")
        
        logger.info(f"Located {len(csv_files)} CSV files for processing")
        return csv_files
    
    def analyze_data_structure(self, csv_files):
        """Analyze the first file to understand data structure"""
        first_file = csv_files[0]
        logger.info(f"Analyzing data structure using {first_file.name}")
        
        # Read a sample to understand structure
        sample_df = pd.read_csv(first_file, nrows=1000, low_memory=False)
        
        # Find label column
        label_candidates = ['Label', ' Label', 'label']
        label_col = None
        
        for candidate in label_candidates:
            if candidate in sample_df.columns:
                label_col = candidate
                break
            # Try stripped version
            stripped_cols = [col.strip() for col in sample_df.columns]
            if candidate.strip() in stripped_cols:
                label_col = sample_df.columns[stripped_cols.index(candidate.strip())]
                break
        
        if label_col is None:
            raise ValueError("Could not identify label column in dataset")
        
        # Get feature columns (all except label)
        feature_cols = [col for col in sample_df.columns if col != label_col]
        
        # Identify numeric features for processing
        numeric_features = sample_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Label column: '{label_col}'")
        logger.info(f"Total features: {len(feature_cols)}")
        logger.info(f"Numeric features: {len(numeric_features)}")
        
        return label_col, numeric_features
    
    def process_chunk(self, chunk, label_col, numeric_features):
        """Processa um chunk de dados, ignorando linhas não numéricas ou headers duplicados, e garantindo sempre as mesmas colunas."""
        if chunk.empty:
            return None, None

        # Detetar e remover headers duplicados (linhas iguais ao header original)
        header_set = set(chunk.columns)
        mask_header = chunk.apply(lambda row: set(str(x).strip() for x in row) == header_set, axis=1)
        n_headers_duplicados = mask_header.sum()
        if n_headers_duplicados > 0:
            chunk = chunk[~mask_header]

        # Garantir que todas as colunas numéricas existem (mesmo que estejam em falta neste chunk)
        for col in numeric_features:
            if col not in chunk.columns:
                chunk[col] = np.nan

        # Converter colunas numéricas, forçando erros a NaN, e garantir ordem das colunas
        X_chunk = chunk[numeric_features].apply(pd.to_numeric, errors='coerce')
        X_chunk = X_chunk[numeric_features]  # garantir ordem
        linhas_invalidas = X_chunk.isnull().all(axis=1)
        n_linhas_invalidas = linhas_invalidas.sum()
        if n_linhas_invalidas > 0:
            chunk = chunk[~linhas_invalidas]
            X_chunk = X_chunk[~linhas_invalidas]

        # Se não restam linhas válidas
        if chunk.empty or X_chunk.empty:
            logger.warning("Chunk vazio após remoção de linhas inválidas/headers duplicados.")
            return None, None

        # Extrair labels (apenas das linhas válidas)
        labels = chunk[label_col]
        is_benign = labels.astype(str).str.upper() == 'BENIGN'
        y_binary = (~is_benign).astype(int)

        # Data cleaning
        X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan)
        X_chunk = X_chunk.fillna(X_chunk.median())

        # Logging do que foi removido
        if n_headers_duplicados > 0 or n_linhas_invalidas > 0:
            logger.warning(f"Chunk: {n_headers_duplicados} headers duplicados e {n_linhas_invalidas} linhas não numéricas removidas.")

        return X_chunk.values, y_binary.values
    
    def load_and_preprocess(self):
        """Load and preprocess the complete CIC-DDoS2019 dataset, garantindo sempre as mesmas colunas."""
        logger.info("Starting CIC-DDoS2019 dataset processing")

        # Find all CSV files
        csv_files = self.find_csv_files()

        # Analyze data structure
        label_col, numeric_features = self.analyze_data_structure(csv_files)

        # Process all files
        all_X_batches = []
        all_y_batches = []
        total_samples = 0

        for file_idx, csv_file in enumerate(csv_files):
            logger.info(f"Processing file {file_idx + 1}/{len(csv_files)}: {csv_file.name}")

            try:
                # Process file in chunks for memory efficiency
                chunk_reader = pd.read_csv(csv_file, chunksize=self.batch_size, low_memory=False)

                for chunk_idx, chunk in enumerate(chunk_reader):
                    X_chunk, y_chunk = self.process_chunk(chunk, label_col, numeric_features)

                    if X_chunk is not None:
                        all_X_batches.append(X_chunk)
                        all_y_batches.append(y_chunk)
                        total_samples += len(chunk)

                    if (chunk_idx + 1) % 10 == 0:
                        logger.info(f"  Processed {(chunk_idx + 1) * self.batch_size:,} rows")

            except Exception as e:
                logger.warning(f"Error processing {csv_file.name}: {str(e)}")
                continue

        if not all_X_batches:
            raise ValueError("No valid data was processed from any file")

        # Combine all processed batches
        logger.info("Combining all processed data")
        X_final = np.vstack(all_X_batches)
        y_final = np.concatenate(all_y_batches)

        # Remover features com variância zero apenas no final
        feature_variance = np.var(X_final, axis=0)
        valid_features_idx = np.where(feature_variance > 0)[0]
        X_final = X_final[:, valid_features_idx]
        final_features = [numeric_features[i] for i in valid_features_idx]

        # Calculate statistics
        benign_count = (y_final == 0).sum()
        attack_count = (y_final == 1).sum()
        attack_ratio = attack_count / len(y_final)

        logger.info(f"Dataset processing complete:")
        logger.info(f"  Total samples: {len(y_final):,}")
        logger.info(f"  Features: {X_final.shape[1]}")
        logger.info(f"  BENIGN samples: {benign_count:,} ({(1-attack_ratio)*100:.1f}%)")
        logger.info(f"  Attack samples: {attack_count:,} ({attack_ratio*100:.1f}%)")

        # Memory cleanup
        del all_X_batches, all_y_batches
        gc.collect()

        return X_final, y_final, final_features
    
    def save_processed_data(self, X, y, feature_names):
        """Save processed data to disk"""
        logger.info("Saving processed CIC-DDoS2019 data")
        
        # Save raw processed data
        np.save(self.output_dir / "X_cic_ddos.npy", X)
        np.save(self.output_dir / "y_cic_ddos.npy", y)
        
        # Create and save feature scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        np.save(self.output_dir / "X_cic_ddos_scaled.npy", X_scaled)
        
        # Save scaler for future use
        with open(self.output_dir / "scaler_cic_ddos.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature names
        with open(self.output_dir / "feature_names_cic_ddos.txt", 'w') as f:
            f.write('\n'.join(feature_names))
        
        # Create and save metadata
        metadata = {
            'dataset': 'CIC-DDoS2019',
            'processing_date': pd.Timestamp.now().isoformat(),
            'total_samples': int(len(y)),
            'feature_count': int(X.shape[1]),
            'benign_samples': int((y == 0).sum()),
            'attack_samples': int((y == 1).sum()),
            'attack_percentage': float((y == 1).mean() * 100),
            'feature_names': feature_names,
            'description': 'CIC-DDoS2019 dataset processed for DDoS detection with BENIGN traffic included'
        }
        
        with open(self.output_dir / "metadata_cic_ddos.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"All data saved to: {self.output_dir}")
        return metadata

def main():
    """Execute the CIC-DDoS2019 processing pipeline"""
    print("CIC-DDoS2019 Dataset Processor")
    print("=" * 50)
    
    try:
        processor = CICDDoSProcessor()
        
        # Verify input directory exists
        if not processor.input_dir.exists():
            print(f"Error: Input directory not found: {processor.input_dir}")
            print("Please ensure the CIC-DDoS2019 dataset is available in the correct location.")
            return False
        
        # Process the dataset
        X, y, feature_names = processor.load_and_preprocess()
        
        # Save processed data
        metadata = processor.save_processed_data(X, y, feature_names)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Dataset: {metadata['dataset']}")
        print(f"Total samples: {metadata['total_samples']:,}")
        print(f"Features: {metadata['feature_count']}")
        print(f"BENIGN traffic: {metadata['benign_samples']:,}")
        print(f"DDoS attacks: {metadata['attack_samples']:,}")
        print(f"Attack ratio: {metadata['attack_percentage']:.1f}%")
        print(f"Output directory: {processor.output_dir}")
        
        logger.info("CIC-DDoS2019 processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        print(f"\nError: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnostico":
        print("Diagnóstico automático de headers dos CSV do CIC-DDoS2019\n" + "="*60)
        diagnosticar_headers_csv()
        exit(0)
    success = main()
    if not success:
        exit(1)
