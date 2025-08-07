#!/usr/bin/env python3
"""
Módulo de Integração de Conjuntos de Dados

Este módulo integra múltiplos conjuntos de dados processados para criar
configurações híbridas para análise abrangente de segurança de rede.
Suporta várias estratégias de integração incluindo aprendizagem ensemble,
conjuntos de dados combinados e configurações de conjuntos de dados individuais.

"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetIntegrator:
    """
    Integrates processed datasets for multi-faceted security analysis.
    
    This class handles the integration of different network security datasets
    to create comprehensive training configurations that can address both
    general intrusion detection and specialized DDoS detection.
    """
    
    def __init__(self, processed_dir=None, output_dir=None):
        """
        Initialize the dataset integrator.
        
        Args:
            processed_dir: Directory containing processed individual datasets
            output_dir: Directory to save integrated dataset configurations
        """
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
    
    def load_dataset(self, dataset_name):
        """Load a processed dataset by name"""
        try:
            X_file = self.processed_dir / f"X_{dataset_name}.npy"
            y_file = self.processed_dir / f"y_{dataset_name}.npy"
            metadata_file = self.processed_dir / f"metadata_{dataset_name}.json"
            
            if not all([X_file.exists(), y_file.exists(), metadata_file.exists()]):
                return None, None, None
            
            X = np.load(X_file)
            y = np.load(y_file)
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded {dataset_name}: {X.shape[0]:,} samples, {X.shape[1]} features")
            return X, y, metadata
            
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {str(e)}")
            return None, None, None
    
    def align_features(self, datasets):
        """
        Align feature dimensions across datasets.
        
        This method handles datasets with different feature counts by either
        padding with zeros or truncating to the minimum common feature set.
        """
        if not datasets:
            return []
        
        # Find minimum feature count
        min_features = min(X.shape[1] for X, _, _ in datasets)
        max_features = max(X.shape[1] for X, _, _ in datasets)
        
        if min_features != max_features:
            logger.info(f"Aligning features: min={min_features}, max={max_features}")
        
        aligned_datasets = []
        for X, y, metadata in datasets:
            if X.shape[1] > min_features:
                # Truncate to minimum features
                X_aligned = X[:, :min_features]
                logger.info(f"Truncated {metadata.get('dataset', 'unknown')} from {X.shape[1]} to {min_features} features")
            elif X.shape[1] < min_features:
                # Pad with zeros
                padding = np.zeros((X.shape[0], min_features - X.shape[1]))
                X_aligned = np.hstack([X, padding])
                logger.info(f"Padded {metadata.get('dataset', 'unknown')} from {X.shape[1]} to {min_features} features")
            else:
                X_aligned = X
            
            aligned_datasets.append((X_aligned, y, metadata))
        
        return aligned_datasets
    
    def create_ensemble_separate_config(self, datasets):
        """
        Create ensemble configuration with separate datasets.
        
        This maintains dataset boundaries for ensemble learning where
        different models can be trained on different datasets.
        """
        logger.info("Creating ensemble separate configuration")
        
        config_data = {}
        total_samples = 0
        
        for i, (X, y, metadata) in enumerate(datasets):
            dataset_key = f"dataset_{i}_{metadata.get('dataset', 'unknown').lower()}"
            
            config_data[dataset_key] = {
                'X': X,
                'y': y,
                'metadata': metadata,
                'samples': X.shape[0],
                'features': X.shape[1]
            }
            
            total_samples += X.shape[0]
        
        # Save ensemble configuration
        output_file = self.output_dir / "ensemble_separate"
        np.savez_compressed(f"{output_file}.npz", **{
            key: data for key, data in config_data.items() 
            if key.endswith(('X', 'y'))
        })
        
        # Save metadata
        ensemble_metadata = {
            'configuration': 'ensemble_separate',
            'description': 'Separate datasets for ensemble learning',
            'total_datasets': len(datasets),
            'total_samples': total_samples,
            'datasets': [data['metadata'] for data in config_data.values()]
        }
        
        with open(f"{output_file}_metadata.json", 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)
        
        logger.info(f"Ensemble separate config saved: {len(datasets)} datasets, {total_samples:,} total samples")
        return ensemble_metadata
    
    def create_combined_config(self, datasets):
        """
        Create combined dataset configuration.
        
        This merges all datasets into a single training set for unified model training.
        """
        logger.info("Creating combined dataset configuration")
        
        # Combine all datasets
        X_combined = np.vstack([X for X, _, _ in datasets])
        y_combined = np.hstack([y for _, y, _ in datasets])
        
        # Shuffle the combined data
        shuffle_idx = np.random.permutation(len(X_combined))
        X_combined = X_combined[shuffle_idx]
        y_combined = y_combined[shuffle_idx]
        
        # Save combined dataset
        output_file = self.output_dir / "combined"
        np.save(f"{output_file}_X.npy", X_combined)
        np.save(f"{output_file}_y.npy", y_combined)
        
        # Calculate combined statistics
        attack_ratio = y_combined.mean()
        dataset_info = []
        
        for _, _, metadata in datasets:
            dataset_info.append({
                'name': metadata.get('dataset', 'unknown'),
                'samples': metadata.get('total_samples', 0),
                'features': metadata.get('feature_count', 0)
            })
        
        combined_metadata = {
            'configuration': 'combined',
            'description': 'All datasets merged into single training set',
            'total_samples': int(len(X_combined)),
            'total_features': int(X_combined.shape[1]),
            'attack_ratio': float(attack_ratio),
            'normal_samples': int((y_combined == 0).sum()),
            'attack_samples': int((y_combined == 1).sum()),
            'source_datasets': dataset_info
        }
        
        with open(f"{output_file}_metadata.json", 'w') as f:
            json.dump(combined_metadata, f, indent=2)
        
        logger.info(f"Combined config saved: {len(X_combined):,} samples, {X_combined.shape[1]} features")
        logger.info(f"Attack ratio: {attack_ratio:.1%}")
        
        return combined_metadata
    
    def create_individual_configs(self, datasets):
        """Create individual dataset configurations for specialized training"""
        logger.info("Creating individual dataset configurations")
        
        individual_metadata = []
        
        for X, y, metadata in datasets:
            dataset_name = metadata.get('dataset', 'unknown').lower().replace('-', '_')
            
            # Save individual dataset
            output_file = self.output_dir / f"individual_{dataset_name}"
            np.save(f"{output_file}_X.npy", X)
            np.save(f"{output_file}_y.npy", y)
            
            # Enhanced metadata for individual use
            individual_meta = {
                'configuration': f'individual_{dataset_name}',
                'description': f'Individual {dataset_name} dataset for specialized training',
                'samples': int(X.shape[0]),
                'features': int(X.shape[1]),
                'attack_ratio': float(y.mean()),
                'source_metadata': metadata
            }
            
            with open(f"{output_file}_metadata.json", 'w') as f:
                json.dump(individual_meta, f, indent=2)
            
            individual_metadata.append(individual_meta)
            
            logger.info(f"Individual {dataset_name} config saved: {X.shape[0]:,} samples")
        
        return individual_metadata
    
    def integrate_datasets(self):
        """Execute the complete dataset integration process"""
        logger.info("Starting dataset integration process")
        
        # Try to load available datasets
        available_datasets = []
        
        # Load NF-UNSW dataset
        X_nf, y_nf, meta_nf = self.load_dataset("nf_unsw")
        if X_nf is not None:
            available_datasets.append((X_nf, y_nf, meta_nf))
        
        # Load CIC-DDoS dataset
        X_cic, y_cic, meta_cic = self.load_dataset("cic_ddos")
        if X_cic is not None:
            available_datasets.append((X_cic, y_cic, meta_cic))
        
        if not available_datasets:
            raise ValueError("No processed datasets found for integration")
        
        logger.info(f"Found {len(available_datasets)} datasets for integration")
        
        # Align features across datasets
        aligned_datasets = self.align_features(available_datasets)
        
        # Create different integration configurations
        configurations = {}
        
        # 1. Ensemble separate configuration
        if len(aligned_datasets) > 1:
            configurations['ensemble_separate'] = self.create_ensemble_separate_config(aligned_datasets)
        
        # 2. Combined configuration
        if len(aligned_datasets) > 1:
            configurations['combined'] = self.create_combined_config(aligned_datasets)
        
        # 3. Individual configurations
        individual_configs = self.create_individual_configs(aligned_datasets)
        for config in individual_configs:
            configurations[config['configuration']] = config
        
        # Save overall integration summary
        integration_summary = {
            'integration_date': pd.Timestamp.now().isoformat(),
            'source_datasets': len(aligned_datasets),
            'configurations_created': list(configurations.keys()),
            'total_samples': sum(X.shape[0] for X, _, _ in aligned_datasets),
            'feature_count': aligned_datasets[0][0].shape[1] if aligned_datasets else 0,
            'configurations': configurations
        }
        
        with open(self.output_dir / "integration_summary.json", 'w') as f:
            json.dump(integration_summary, f, indent=2)
        
        logger.info("Dataset integration completed successfully")
        logger.info(f"Created {len(configurations)} configurations")
        logger.info(f"Output directory: {self.output_dir}")
        
        return integration_summary

def main():
    """Execute the dataset integration pipeline"""
    print("Dataset Integration Module")
    print("=" * 50)
    
    try:
        integrator = DatasetIntegrator()
        
        # Check if processed data exists
        if not integrator.processed_dir.exists():
            print(f"Error: Processed data directory not found: {integrator.processed_dir}")
            print("Please run individual dataset processors first.")
            return False
        
        # Execute integration
        summary = integrator.integrate_datasets()
        
        # Print results
        print("\nIntegration Summary:")
        print(f"Source datasets: {summary['source_datasets']}")
        print(f"Total samples: {summary['total_samples']:,}")
        print(f"Feature count: {summary['feature_count']}")
        print(f"Configurations created: {len(summary['configurations_created'])}")
        
        for config_name in summary['configurations_created']:
            print(f"  - {config_name}")
        
        print(f"Output directory: {integrator.output_dir}")
        
        logger.info("Dataset integration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Integration failed: {str(e)}")
        print(f"\nError: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
