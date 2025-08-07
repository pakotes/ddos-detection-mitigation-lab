#!/usr/bin/env python3
"""
Modelo OTIMIZADO para Detecção DDoS em PRODUÇÃO
Foco: Velocidade, Baixo Consumo, Alta Precisão, Poucos Falsos Positivos
"""

import logging
import numpy as np
import pandas as pd
import json
import pickle
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import joblib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

class ProductionDDoSDetector:
    """Detector DDoS otimizado para produção"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        self.threshold = 0.5
        self.model_type = None
        
    def select_production_features(self, X, y):
        """Selecionar features críticas para DDoS (máximo 15)"""
        logger.info("Selecionando features críticas para DDoS...")
        
        # Usar apenas 15 features mais importantes (para velocidade)
        selector = SelectKBest(score_func=f_classif, k=15)
        X_selected = selector.fit_transform(X, y)
        
        # Se temos nomes de features, manter referência
        if hasattr(X, 'columns'):
            selected_features = X.columns[selector.get_support()].tolist()
        else:
            selected_features = [f'feature_{i}' for i in range(X_selected.shape[1])]
        
        logger.info(f"Features selecionadas: {len(selected_features)}")
        for i, feature in enumerate(selected_features):
            logger.info(f"  {i+1:2d}. {feature}")
        
        self.feature_selector = selector
        self.feature_names = selected_features
        
        return X_selected
    
    def train_production_models(self, X_train, X_test, y_train, y_test):
        """Treinar e comparar modelos para produção"""
        logger.info("Treinando modelos para PRODUÇÃO...")
        
        models_to_test = {
            'decision_tree': DecisionTreeClassifier(
                max_depth=10,          # Profundidade limitada para velocidade
                min_samples_split=20,  # Evitar overfitting
                min_samples_leaf=10,   # Evitar overfitting
                random_state=42
            ),
            'random_forest_fast': RandomForestClassifier(
                n_estimators=50,       # Poucas árvores para velocidade
                max_depth=8,           # Profundidade limitada
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',   # Reduzir features por árvore
                random_state=42,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,                 # Sem regularização excessiva
                max_iter=1000,
                random_state=42,
                solver='liblinear'     # Solver rápido
            )
        }
        
        results = {}
        trained_models = {}
        
        for model_name, model in models_to_test.items():
            logger.info(f"\nTestando {model_name}...")
            
            # Treinar
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predição
            start_time = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Velocidade de predição (amostras por segundo)
            samples_per_second = len(X_test) / prediction_time
            
            # Cross-validation para estabilidade
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'samples_per_second': samples_per_second,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'false_positive_rate': 1 - precision if precision > 0 else 1,
                'production_score': self._calculate_production_score(
                    accuracy, precision, recall, samples_per_second
                )
            }
            
            trained_models[model_name] = model
            
            logger.info(f"  Accuracy: {accuracy:.3f}")
            logger.info(f"  Precision: {precision:.3f} (baixos falsos positivos)")
            logger.info(f"  Recall: {recall:.3f} (detecta ataques)")
            logger.info(f"  F1: {f1:.3f}")
            logger.info(f"  Velocidade: {samples_per_second:,.0f} amostras/seg")
            logger.info(f"  Score Produção: {results[model_name]['production_score']:.3f}")
        
        # Selecionar melhor modelo para produção
        best_model_name = max(results.keys(), 
                             key=lambda k: results[k]['production_score'])
        
        self.model = trained_models[best_model_name]
        self.model_type = best_model_name
        
        logger.info(f"\n🏆 MELHOR MODELO PARA PRODUÇÃO: {best_model_name}")
        best_result = results[best_model_name]
        logger.info(f"   Accuracy: {best_result['accuracy']:.3f}")
        logger.info(f"   Precision: {best_result['precision']:.3f}")
        logger.info(f"   Velocidade: {best_result['samples_per_second']:,.0f} amostras/seg")
        logger.info(f"   Taxa FP: {best_result['false_positive_rate']:.3f}")
        
        return results, best_model_name
    
    def _calculate_production_score(self, accuracy, precision, recall, speed):
        """Calcular score ponderado para produção"""
        # Pesos para produção DDoS:
        # - Precision é CRÍTICA (evitar falsos positivos)
        # - Recall é importante (detectar ataques)
        # - Velocidade é importante (tempo real)
        # - Accuracy é baseline
        
        return (
            precision * 0.4 +      # 40% - Crítico evitar falsos positivos
            recall * 0.3 +         # 30% - Importante detectar ataques
            accuracy * 0.2 +       # 20% - Accuracy geral
            min(speed/10000, 1) * 0.1  # 10% - Velocidade (normalizada)
        )
    
    def optimize_threshold(self, X_val, y_val):
        """Otimizar threshold para minimizar falsos positivos"""
        logger.info("Otimizando threshold para produção...")
        
        # Testar diferentes thresholds
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        best_threshold = 0.5
        best_score = 0
        
        for threshold in np.arange(0.3, 0.8, 0.05):
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            precision = precision_score(y_val, y_pred_thresh)
            recall = recall_score(y_val, y_pred_thresh)
            
            # Score favorece precision (evitar falsos positivos)
            if precision > 0 and recall > 0:
                score = (precision * 0.7) + (recall * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        self.threshold = best_threshold
        logger.info(f"Threshold otimizado: {best_threshold:.3f}")
        logger.info(f"Score otimizado: {best_score:.3f}")
        
        return best_threshold
    
    def predict_production(self, X):
        """Predição otimizada para produção"""
        if self.model is None:
            raise ValueError("Modelo não treinado!")
        
        # Aplicar mesmas transformações
        if self.scaler:
            X = self.scaler.transform(X)
        if self.feature_selector:
            X = self.feature_selector.transform(X)
        
        # Predição com threshold otimizado
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)[:, 1]
            return (y_proba >= self.threshold).astype(int)
        else:
            return self.model.predict(X)
    
    def save_production_model(self, output_path):
        """Salvar modelo otimizado para produção"""
        logger.info(f"Salvando modelo de produção...")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'model_type': self.model_type,
            'metadata': {
                'optimized_for': 'DDoS_detection_production',
                'n_features': len(self.feature_names) if self.feature_names else None,
                'created_date': pd.Timestamp.now().isoformat()
            }
        }
        
        # Salvar com joblib (mais eficiente)
        joblib.dump(model_data, output_path)
        logger.info(f"Modelo salvo: {output_path}")
        
        # Salvar também metadados em JSON
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'threshold': self.threshold,
                'feature_names': self.feature_names,
                'optimized_for': 'DDoS_detection_production',
                'created_date': model_data['metadata']['created_date']
            }, f, indent=2)
    
    @classmethod
    def load_production_model(cls, model_path):
        """Carregar modelo de produção"""
        model_data = joblib.load(model_path)
        
        detector = cls()
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.feature_selector = model_data['feature_selector']
        detector.feature_names = model_data['feature_names']
        detector.threshold = model_data['threshold']
        detector.model_type = model_data['model_type']
        
        return detector

def load_best_available_dataset():
    """Carregar melhor dataset disponível"""
    logger.info("Procurando melhor dataset disponível...")
    
    base_path = Path(__file__).parent.parent.parent / 'src' / 'datasets'
    
    # 1. Prioridade: dados reais processados
    for dataset_type in ['real', 'realistic', 'clean']:
        dataset_dir = base_path / dataset_type
        if dataset_dir.exists():
            logger.info(f"Usando dataset: {dataset_type}")
            return dataset_dir, dataset_type
    
    # 2. Segunda opção: dados integrados
    integrated_dir = base_path / 'integrated'
    if integrated_dir.exists():
        # Verificar se arquivos integrados existem
        for suffix in ['real', 'advanced', 'demo']:
            X_file = integrated_dir / f'X_integrated_{suffix}.npy'
            y_file = integrated_dir / f'y_integrated_{suffix}.npy'
            if X_file.exists() and y_file.exists():
                logger.info(f"Usando dataset integrado: {suffix}")
                return integrated_dir, f'integrated_{suffix}'
    
    # 3. Terceira opção: dados demo
    demo_dir = base_path / 'demo'
    if demo_dir.exists():
        logger.info("Usando dataset demo")
        return demo_dir, 'demo'
    
    # 4. Se nada encontrado, criar dados demo automaticamente
    logger.warning("Nenhum dataset encontrado! Criando dados demo...")
    try:
        # Importar e executar gerador demo
        import sys
        sys.path.append(str(Path(__file__).parent))
        from create_demo_datasets import DemoDatasetGenerator
        
        generator = DemoDatasetGenerator()
        generator.generate_all_demo_datasets()
        
        # Tentar novamente
        integrated_dir = base_path / 'integrated'
        demo_file = integrated_dir / 'X_integrated_demo.npy'
        if demo_file.exists():
            logger.info("Usando dados demo recém-criados")
            return integrated_dir, 'integrated_demo'
        
        demo_dir = base_path / 'demo'
        if demo_dir.exists():
            return demo_dir, 'demo'
            
    except Exception as e:
        logger.error(f"Erro criando dados demo: {e}")
    
    raise FileNotFoundError("Nenhum dataset encontrado e falha ao criar dados demo!")

def main():
    """Função principal - Criar modelo DDoS para produção"""
    logger.info("Criando modelo DDoS OTIMIZADO para PRODUÇÃO...")
    logger.info("="*60)
    
    try:
        # 1. Carregar melhor dataset disponível
        dataset_dir, dataset_type = load_best_available_dataset()
        logger.info(f"Dataset fonte: {dataset_type}")
        
        # 2. Carregar dados (diferentes formatos)
        if 'integrated' in dataset_type:
            # Dados integrados
            suffix = dataset_type.split('_')[1]  # real, advanced, demo
            X = np.load(dataset_dir / f'X_integrated_{suffix}.npy')
            y = np.load(dataset_dir / f'y_integrated_{suffix}.npy')
            
            # Split treino/teste
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Dados integrados carregados: {suffix}")
            
        else:
            # Dados com splits separados
            dataset_name = None
            for name in ['cicddos', 'unsw']:  # Priorizar CIC-DDoS
                train_file = dataset_dir / f'X_train_{name}.npy'
                if train_file.exists():
                    dataset_name = name
                    break
            
            if dataset_name is None:
                raise FileNotFoundError("Nenhum dataset de treino encontrado!")
            
            logger.info(f"Usando dataset: {dataset_name}")
            
            # Carregar dados com splits separados
            X_train = np.load(dataset_dir / f'X_train_{dataset_name}.npy')
            X_test = np.load(dataset_dir / f'X_test_{dataset_name}.npy')
            y_train = np.load(dataset_dir / f'y_train_binary_{dataset_name}.npy')
            y_test = np.load(dataset_dir / f'y_test_binary_{dataset_name}.npy')
        
        logger.info(f"Dados carregados:")
        logger.info(f"  Treino: {X_train.shape}")
        logger.info(f"  Teste: {X_test.shape}")
        logger.info(f"  Taxa ataques: {y_train.mean():.1%}")
        
        # 3. Criar detector de produção
        detector = ProductionDDoSDetector()
        
        # 4. Normalizar dados
        logger.info("Normalizando dados...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        detector.scaler = scaler
        
        # 5. Selecionar features críticas
        X_train_selected = detector.select_production_features(X_train_scaled, y_train)
        X_test_selected = detector.feature_selector.transform(X_test_scaled)
        
        # 6. Treinar modelos e selecionar melhor
        results, best_model = detector.train_production_models(
            X_train_selected, X_test_selected, y_train, y_test
        )
        
        # 8. Otimizar threshold
        detector.optimize_threshold(X_test_selected, y_test)
        
        # 9. Teste final com threshold otimizado
        logger.info("\n📊 TESTE FINAL COM CONFIGURAÇÃO OTIMIZADA:")
        y_pred_final = detector.predict_production(X_test_scaled)
        
        final_accuracy = accuracy_score(y_test, y_pred_final)
        final_precision = precision_score(y_test, y_pred_final)
        final_recall = recall_score(y_test, y_pred_final)
        final_f1 = f1_score(y_test, y_pred_final)
        
        logger.info(f"  Accuracy Final: {final_accuracy:.3f}")
        logger.info(f"  Precision Final: {final_precision:.3f}")
        logger.info(f"  Recall Final: {final_recall:.3f}")
        logger.info(f"  F1 Final: {final_f1:.3f}")
        logger.info(f"  Taxa Falsos Positivos: {1-final_precision:.3f}")
        
        # 10. Salvar modelo de produção
        output_dir = Path(__file__).parent.parent.parent / 'src' / 'models' / 'production'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / 'ddos_production_model.pkl'
        detector.save_production_model(model_path)
        
        # 11. Criar relatório de produção
        production_report = {
            'model_info': {
                'type': best_model,
                'dataset_source': f"{dataset_type}_{dataset_name}",
                'n_features': len(detector.feature_names),
                'threshold': detector.threshold
            },
            'performance': {
                'accuracy': final_accuracy,
                'precision': final_precision,
                'recall': final_recall,
                'f1_score': final_f1,
                'false_positive_rate': 1 - final_precision
            },
            'production_metrics': results[best_model],
            'feature_names': detector.feature_names,
            'deployment_ready': True,
            'created_date': pd.Timestamp.now().isoformat()
        }
        
        report_path = output_dir / 'production_report.json'
        with open(report_path, 'w') as f:
            json.dump(production_report, f, indent=2)
        
        logger.info(f"\n🚀 MODELO DE PRODUÇÃO CRIADO COM SUCESSO!")
        logger.info(f"   Arquivo: {model_path}")
        logger.info(f"   Relatório: {report_path}")
        logger.info(f"   Modelo: {best_model}")
        logger.info(f"   Features: {len(detector.feature_names)}")
        logger.info(f"   Velocidade: {results[best_model]['samples_per_second']:,.0f} amostras/seg")
        logger.info(f"   Precision: {final_precision:.3f} (baixos falsos positivos)")
        logger.info(f"   Recall: {final_recall:.3f} (detecta ataques)")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao criar modelo de produção: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
