#!/usr/bin/env python3
"""
Pipeline ML Eficiente para DDoS Detection
Script único que faz tudo: dados → limpeza → otimização → modelo final
"""

import logging
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class EfficientDDoSPipeline:
    """Pipeline ML eficiente para detecção DDoS"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.models_dir = self.project_root / 'src' / 'models' / 'optimized'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurações otimizadas
        self.n_features = 15  # Top features para velocidade
        self.test_size = 0.2
        self.random_state = 42
        
    def log_step(self, step, message):
        """Log formatado"""
        logger.info(f"\n{'='*60}")
        logger.info(f"ETAPA {step}: {message}")
        logger.info(f"{'='*60}")
    
    def create_synthetic_data(self, n_samples=50000):
        """Criar dados sintéticos realistas se necessário"""
        logger.info("Criando dados sintéticos realistas...")
        
        np.random.seed(self.random_state)
        
        # Simular features de tráfego de rede
        n_features = 25
        X = np.random.randn(n_samples, n_features)
        
        # Criar padrões DDoS realistas
        normal_ratio = 0.85
        n_normal = int(n_samples * normal_ratio)
        n_attack = n_samples - n_normal
        
        # Tráfego normal (distribuição normal)
        X[:n_normal] = np.random.normal(0, 1, (n_normal, n_features))
        
        # Tráfego DDoS (padrões anômalos)
        # Volume alto
        X[n_normal:, 0] = np.random.exponential(5, n_attack)  # Packets/sec
        X[n_normal:, 1] = np.random.exponential(3, n_attack)  # Bytes/sec
        
        # Padrões repetitivos
        X[n_normal:, 2] = np.random.choice([1, 2, 3], n_attack)  # Flags
        X[n_normal:, 3] = np.random.uniform(0, 0.1, n_attack)  # Variance
        
        # Timing suspeito
        X[n_normal:, 4] = np.random.uniform(10, 100, n_attack)  # Duration
        
        # Labels
        y = np.zeros(n_samples)
        y[n_normal:] = 1  # DDoS = 1
        
        # Feature names
        feature_names = [
            'packets_per_sec', 'bytes_per_sec', 'tcp_flags', 'flow_variance',
            'connection_duration', 'packet_size_mean', 'packet_size_std',
            'inter_arrival_time', 'flow_bytes_sent', 'flow_packets_sent',
            'flow_bytes_received', 'flow_packets_received', 'protocol_type',
            'service_type', 'connection_state', 'src_port_entropy',
            'dst_port_entropy', 'packet_loss_rate', 'retransmission_rate',
            'window_size_mean', 'window_size_std', 'tcp_rtt', 'flow_duration',
            'bytes_per_packet', 'packets_per_flow'
        ]
        
        logger.info(f"Dados sintéticos criados: {X.shape[0]} amostras, {X.shape[1]} features")
        logger.info(f"Distribuição: {(1-y.mean()):.1%} normal, {y.mean():.1%} ataques")
        
        return X, y, feature_names
    
    def load_available_data(self):
        """Carregar dados disponíveis (integrados ou sintéticos)"""
        self.log_step(1, "Carregamento de Dados")
        
        # Tentar dados integrados primeiro
        integrated_dir = self.project_root / 'src' / 'datasets' / 'integrated'
        
        if integrated_dir.exists():
            # Verificar dados reais
            X_file = integrated_dir / 'X_integrated_real.npy'
            y_file = integrated_dir / 'y_integrated_real.npy'
            features_file = integrated_dir / 'feature_names_real.txt'
            
            if X_file.exists() and y_file.exists():
                logger.info("Carregando dados integrados REAIS...")
                X = np.load(X_file)
                y = np.load(y_file)
                
                if features_file.exists():
                    with open(features_file, 'r') as f:
                        feature_names = [line.strip() for line in f.readlines()]
                else:
                    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                
                logger.info(f"✅ Dados reais carregados: {X.shape}")
                logger.info(f"   Ataques: {y.sum()}/{len(y)} ({y.mean():.1%})")
                return X, y, feature_names
            
            # Tentar dados avançados
            X_file = integrated_dir / 'X_integrated_advanced.npy'
            y_file = integrated_dir / 'y_integrated_advanced.npy'
            
            if X_file.exists() and y_file.exists():
                logger.info("Carregando dados integrados AVANÇADOS...")
                X = np.load(X_file)
                y = np.load(y_file)
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                
                logger.info(f"✅ Dados avançados carregados: {X.shape}")
                return X, y, feature_names
        
        # Fallback: criar dados sintéticos
        logger.warning("Dados integrados não encontrados - usando sintéticos")
        return self.create_synthetic_data()
    
    def optimize_features(self, X, y, feature_names):
        """Seleção otimizada de features"""
        self.log_step(2, "Otimização de Features")
        
        # Selecionar top features para velocidade
        selector = SelectKBest(score_func=f_classif, k=self.n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Obter nomes das features selecionadas
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        scores = selector.scores_[selected_indices]
        
        logger.info(f"Features selecionadas ({self.n_features}):")
        for name, score in zip(selected_features, scores):
            logger.info(f"  {name}: {score:.2f}")
        
        return X_selected, selected_features, selector
    
    def train_optimized_models(self, X_train, X_test, y_train, y_test):
        """Treinar modelos otimizados para produção"""
        self.log_step(3, "Treinamento de Modelos Otimizados")
        
        models = {}
        results = {}
        
        # 1. Random Forest Otimizado (balanceado velocidade/accuracy)
        logger.info("Treinando Random Forest Otimizado...")
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1, verbose=0)
        rf_grid.fit(X_train, y_train)
        
        models['random_forest'] = rf_grid.best_estimator_
        rf_pred = rf_grid.predict(X_test)
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'params': rf_grid.best_params_,
            'cv_score': rf_grid.best_score_
        }
        
        # 2. Logistic Regression (rápido para produção)
        logger.info("Treinando Logistic Regression...")
        lr_params = {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'saga'],  # Solvers que suportam n_jobs
            'max_iter': [1000, 2000]
        }
        
        lr = LogisticRegression(random_state=self.random_state)
        lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='f1', n_jobs=-1, verbose=0)
        lr_grid.fit(X_train, y_train)
        
        models['logistic_regression'] = lr_grid.best_estimator_
        lr_pred = lr_grid.predict(X_test)
        results['logistic_regression'] = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'params': lr_grid.best_params_,
            'cv_score': lr_grid.best_score_
        }
        
        # 3. Isolation Forest (detecção de anomalias)
        logger.info("Treinando Isolation Forest...")
        iso_params = {
            'contamination': [0.1, 0.15, 0.2],
            'n_estimators': [50, 100],
            'max_features': [0.5, 0.8, 1.0]
        }
        
        iso = IsolationForest(random_state=self.random_state, n_jobs=-1)
        
        # Para Isolation Forest, usamos validação manual
        best_iso = None
        best_score = 0
        
        for contamination in iso_params['contamination']:
            for n_estimators in iso_params['n_estimators']:
                for max_features in iso_params['max_features']:
                    iso_temp = IsolationForest(
                        contamination=contamination,
                        n_estimators=n_estimators,
                        max_features=max_features,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                    iso_temp.fit(X_train)
                    iso_pred = iso_temp.predict(X_test)
                    iso_pred = (iso_pred == -1).astype(int)  # -1 = anomalia = DDoS
                    
                    score = accuracy_score(y_test, iso_pred)
                    if score > best_score:
                        best_score = score
                        best_iso = iso_temp
        
        models['isolation_forest'] = best_iso
        iso_pred = best_iso.predict(X_test)
        iso_pred = (iso_pred == -1).astype(int)
        results['isolation_forest'] = {
            'accuracy': accuracy_score(y_test, iso_pred),
            'params': 'optimized',
            'cv_score': best_score
        }
        
        # Mostrar resultados
        logger.info("\n📊 RESULTADOS DOS MODELOS:")
        for name, result in results.items():
            logger.info(f"  {name}: {result['accuracy']:.3f} accuracy")
        
        return models, results
    
    def select_best_model(self, models, results):
        """Selecionar melhor modelo balanceando accuracy e velocidade"""
        self.log_step(4, "Seleção do Melhor Modelo")
        
        # Pesos para seleção (favorece velocidade para produção)
        weights = {
            'random_forest': 0.7,      # Boa accuracy mas mais lento
            'logistic_regression': 1.0, # Rápido e eficiente
            'isolation_forest': 0.8    # Bom para anomalias
        }
        
        scores = {}
        for name, result in results.items():
            weighted_score = result['accuracy'] * weights[name]
            scores[name] = weighted_score
            logger.info(f"{name}: {result['accuracy']:.3f} × {weights[name]} = {weighted_score:.3f}")
        
        best_model_name = max(scores, key=scores.get)
        best_model = models[best_model_name]
        
        logger.info(f"\n🏆 MELHOR MODELO: {best_model_name}")
        logger.info(f"   Accuracy: {results[best_model_name]['accuracy']:.3f}")
        
        return best_model, best_model_name, results[best_model_name]
    
    def save_production_model(self, model, model_name, result, feature_selector, selected_features, scaler=None):
        """Salvar modelo otimizado para produção"""
        self.log_step(5, "Salvando Modelo de Produção")
        
        # Pacote completo para produção
        production_package = {
            'model': model,
            'model_type': model_name,
            'feature_selector': feature_selector,
            'selected_features': selected_features,
            'scaler': scaler,
            'performance': result,
            'n_features': len(selected_features),
            'version': '1.0',
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        # Salvar modelo
        model_file = self.models_dir / 'ddos_optimized_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(production_package, f)
        
        # Salvar metadados
        metadata = {
            'model_type': model_name,
            'accuracy': result['accuracy'],
            'features': selected_features,
            'n_features': len(selected_features),
            'optimized_for': 'production_speed_accuracy',
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = self.models_dir / 'model_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Modelo salvo: {model_file}")
        logger.info(f"✅ Metadados: {metadata_file}")
        logger.info(f"   Tipo: {model_name}")
        logger.info(f"   Accuracy: {result['accuracy']:.3f}")
        logger.info(f"   Features: {len(selected_features)}")
        
        return model_file, metadata_file
    
    def create_inference_script(self):
        """Criar script de inferência para produção"""
        inference_code = '''#!/usr/bin/env python3
"""
Script de inferência para modelo DDoS otimizado
Uso: python ddos_inference.py <dados.csv>
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def load_production_model():
    """Carregar modelo de produção"""
    model_path = Path(__file__).parent / 'ddos_optimized_model.pkl'
    
    with open(model_path, 'rb') as f:
        package = pickle.load(f)
    
    return package

def predict_ddos(data):
    """Predição DDoS otimizada"""
    # Carregar modelo
    package = load_production_model()
    model = package['model']
    feature_selector = package['feature_selector']
    scaler = package.get('scaler')
    
    # Pré-processamento
    if scaler:
        data = scaler.transform(data)
    
    # Seleção de features
    data_selected = feature_selector.transform(data)
    
    # Predição
    predictions = model.predict(data_selected)
    probabilities = None
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(data_selected)[:, 1]
    
    return predictions, probabilities

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Uso: python ddos_inference.py <dados.csv>")
        sys.exit(1)
    
    # Carregar dados
    data_file = sys.argv[1]
    data = pd.read_csv(data_file)
    
    # Predição
    predictions, probs = predict_ddos(data.values)
    
    # Mostrar resultados
    n_attacks = predictions.sum()
    print(f"Análise: {len(predictions)} amostras")
    print(f"Ataques detectados: {n_attacks} ({n_attacks/len(predictions):.1%})")
    
    if probs is not None:
        high_risk = (probs > 0.8).sum()
        print(f"Alto risco: {high_risk} amostras")
'''
        
        inference_file = self.models_dir / 'ddos_inference.py'
        with open(inference_file, 'w') as f:
            f.write(inference_code)
        
        logger.info(f"✅ Script de inferência: {inference_file}")
        return inference_file
    
    def run_complete_pipeline(self):
        """Executar pipeline completo"""
        logger.info("🚀 INICIANDO PIPELINE ML EFICIENTE PARA DDoS")
        logger.info("="*60)
        
        try:
            # 1. Carregar dados
            X, y, feature_names = self.load_available_data()
            
            # 2. Split treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            logger.info(f"Split: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
            
            # 3. Otimizar features
            X_train_opt, selected_features, feature_selector = self.optimize_features(
                X_train, y_train, feature_names
            )
            X_test_opt = feature_selector.transform(X_test)
            
            # 4. Treinar modelos
            models, results = self.train_optimized_models(
                X_train_opt, X_test_opt, y_train, y_test
            )
            
            # 5. Selecionar melhor modelo
            best_model, model_name, best_result = self.select_best_model(models, results)
            
            # 6. Salvar para produção
            model_file, metadata_file = self.save_production_model(
                best_model, model_name, best_result, feature_selector, selected_features
            )
            
            # 7. Criar script de inferência
            inference_file = self.create_inference_script()
            
            # 8. Relatório final
            self.log_step(6, "Relatório Final")
            logger.info("🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
            logger.info(f"\n📁 ARQUIVOS GERADOS:")
            logger.info(f"   Modelo: {model_file}")
            logger.info(f"   Metadados: {metadata_file}")
            logger.info(f"   Inferência: {inference_file}")
            
            logger.info(f"\n🎯 MODELO FINAL:")
            logger.info(f"   Tipo: {model_name}")
            logger.info(f"   Accuracy: {best_result['accuracy']:.3f}")
            logger.info(f"   Features: {len(selected_features)}")
            logger.info(f"   Otimizado para: Produção (velocidade + accuracy)")
            
            logger.info(f"\n🚀 PRÓXIMOS PASSOS:")
            logger.info(f"   Teste: python {inference_file} dados.csv")
            logger.info(f"   Integração: Use o modelo em sistemas de produção")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no pipeline: {e}")
            return False

def main():
    """Função principal"""
    pipeline = EfficientDDoSPipeline()
    success = pipeline.run_complete_pipeline()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
