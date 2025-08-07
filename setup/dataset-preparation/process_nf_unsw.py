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
    """Processa o conjunto de dados NF-UNSW-NB15-v3 para detecção de intrusões"""
    
    def __init__(self, input_dir=None, output_dir=None):
        if input_dir is None:
            # Use local dataset directory
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
        """Load and preprocess NF-UNSW-NB15-v3 data"""
        logger.info("Starting NF-UNSW-NB15-v3 processing...")
        
        # Find CSV files
        csv_files = list(self.input_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.input_dir}")
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        dataframes = []
        for csv_file in csv_files:
            logger.info(f"Loading {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                if not df.empty:
                    dataframes.append(df)
                    logger.info(f"   Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
            except Exception as e:
                logger.warning(f"   Error loading {csv_file.name}: {e}")
        
        if not dataframes:
            raise ValueError("No valid files loaded")
        
        # Combine datasets
        logger.info("Combining datasets...")
        df_combined = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined dataset: {df_combined.shape[0]:,} rows, {df_combined.shape[1]} columns")
        
        return self._preprocess_features(df_combined)
    
    def _preprocess_features(self, df):
        """Preprocess dataset features"""
        logger.info("Preprocessing features...")
        
        # Identify label column
        label_candidates = ['Label', ' Label', 'label', 'attack', 'Attack']
        label_col = None
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("Label column not found")
        
        logger.info(f"Using label column: {label_col}")
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col != label_col]
        
        # Process labels
        y_raw = df[label_col]
        if y_raw.dtype == 'object':
            # Categorical labels
            normal_labels = ['BENIGN', 'Normal', 'normal', 'NORMAL']
            y_binary = (~y_raw.isin(normal_labels)).astype(int)
            logger.info("Converted categorical labels to binary")
        else:
            # Numeric labels
            y_binary = (y_raw != 0).astype(int)
            logger.info("Processed numeric labels")
        
        # Process features
        X = df[feature_cols].copy()
        
        # Select only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].copy()
        
        # Process simple categorical columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            logger.info(f"Processing {len(categorical_cols)} categorical columns...")
            for col in categorical_cols:
                unique_count = X[col].nunique()
                if unique_count < 50:  # Reasonable limit
                    try:
                        le = LabelEncoder()
                        X_numeric[f"{col}_encoded"] = le.fit_transform(X[col].fillna('unknown'))
                        logger.info(f"   Encoded {col} ({unique_count} values)")
                    except:
                        logger.warning(f"   Error encoding {col}")
        
        # Clean data
        logger.info("Cleaning data...")
        
        # Handle infinities
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values
        missing_count = X_numeric.isnull().sum().sum()
        if missing_count > 0:
            X_numeric = X_numeric.fillna(X_numeric.median())
            logger.info(f"   Filled {missing_count} missing values")
        
        # Remove zero variance features
        variance = X_numeric.var()
        zero_var_cols = variance[variance == 0].index
        if len(zero_var_cols) > 0:
            X_numeric = X_numeric.drop(columns=zero_var_cols)
            logger.info(f"   Removed {len(zero_var_cols)} zero variance features")
        
        # Final statistics
        attack_count = y_binary.sum()
        normal_count = len(y_binary) - attack_count
        attack_ratio = attack_count / len(y_binary)
        
        logger.info(f"Final dataset: {X_numeric.shape[0]:,} samples, {X_numeric.shape[1]} features")
        logger.info(f"Normal: {normal_count:,} ({1-attack_ratio:.1%})")
        logger.info(f"Attacks: {attack_count:,} ({attack_ratio:.1%})")
        
        return X_numeric.values, y_binary.values, list(X_numeric.columns)
    
    def save_processed_data(self, X, y, feature_names):
        """Save processed data to disk"""
        logger.info("Saving processed data...")
        
        # Save processed arrays
        np.save(self.output_dir / "X_nf_unsw.npy", X)
        np.save(self.output_dir / "y_nf_unsw.npy", y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaled data
        np.save(self.output_dir / "X_nf_unsw_scaled.npy", X_scaled)
        
        # Save scaler
        with open(self.output_dir / "scaler_nf_unsw.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature names
        with open(self.output_dir / "feature_names_nf_unsw.txt", 'w') as f:
            f.write('\n'.join(feature_names))
        
        # Save metadata
        metadata = {
            'dataset': 'NF-UNSW-NB15-v3',
            'samples': int(X.shape[0]),
            'features': int(X.shape[1]),
            'attack_samples': int(y.sum()),
            'normal_samples': int(len(y) - y.sum()),
            'attack_ratio': float(y.sum() / len(y)),
            'feature_names': feature_names
        }
        
        with open(self.output_dir / "metadata_nf_unsw.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Data saved to {self.output_dir}")
        logger.info(f"Total samples: {X.shape[0]:,}")
        logger.info(f"Features: {X.shape[1]}")
        
        return metadata

def main():
    """Execute the NF-UNSW-NB15-v3 processing pipeline"""
    try:
        processor = NFUNSWProcessor()
        
        # Check if input directory exists
        if not processor.input_dir.exists():
            logger.error(f"Input directory not found: {processor.input_dir}")
            logger.info("Please ensure NF-UNSW-NB15-v3 dataset is available")
            return False
        
        # Process data
        X, y, feature_names = processor.load_and_preprocess()
        
        # Save processed data
        metadata = processor.save_processed_data(X, y, feature_names)
        
        logger.info("NF-UNSW-NB15-v3 processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False

if __name__ == "__main__":
    main()
