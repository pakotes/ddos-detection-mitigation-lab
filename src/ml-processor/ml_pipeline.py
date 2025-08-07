"""
Hybrid ML Pipeline for DDoS Detection

This module implements the core machine learning pipeline developed during
my master's thesis research. It combines supervised and unsupervised learning
approaches to improve DDoS detection accuracy while reducing false positives.

Key components:
1. Multi-source data consumption (Suricata, NetFlow)
2. Hybrid ML models (XGBoost + Isolation Forest)
3. Reputation-based scoring system
4. Real-time threat intelligence

Research Context:
- Master's thesis in Network Engineering
- Focus on BGP-based cooperative blocking
- Performance target: <200ms detection latency
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pickle
import redis
from kafka import KafkaConsumer, KafkaProducer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from dataclasses import dataclass
import structlog

# Configuração de logging estruturado
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@dataclass
class ThreatScore:
    """Classe para representar o score de ameaça de um IP"""
    ip: str
    score: float
    attack_type: str
    confidence: float
    detection_method: str
    timestamp: datetime
    ttl: int = 3600  # Time to live em segundos


class FeatureExtractor:
    """Extrator de características para análise de tráfego"""
    
    def extract_flow_features(self, flow_data: Dict) -> Dict:
        """Extrai características de dados de fluxo NetFlow/sFlow"""
        features = {
            # Características básicas de fluxo
            'bytes_per_second': flow_data.get('bytes', 0) / max(flow_data.get('duration', 1), 1),
            'packets_per_second': flow_data.get('packets', 0) / max(flow_data.get('duration', 1), 1),
            'avg_packet_size': flow_data.get('bytes', 0) / max(flow_data.get('packets', 1), 1),
            
            # Características de protocolo
            'protocol': flow_data.get('protocol', 0),
            'src_port': flow_data.get('src_port', 0),
            'dst_port': flow_data.get('dst_port', 0),
            
            # Flags TCP
            'tcp_flags': flow_data.get('tcp_flags', 0),
            'syn_flag': 1 if flow_data.get('tcp_flags', 0) & 0x02 else 0,
            'ack_flag': 1 if flow_data.get('tcp_flags', 0) & 0x10 else 0,
            'fin_flag': 1 if flow_data.get('tcp_flags', 0) & 0x01 else 0,
            'rst_flag': 1 if flow_data.get('tcp_flags', 0) & 0x04 else 0,
            
            # Duração e timing
            'flow_duration': flow_data.get('duration', 0),
            'inter_arrival_time': flow_data.get('inter_arrival_time', 0),
        }
        
        return features
    
    def extract_suricata_features(self, suricata_data: Dict) -> Dict:
        """Extrai características de logs do Suricata"""
        features = {
            # Alert information
            'alert_severity': suricata_data.get('alert', {}).get('severity', 0),
            'alert_category': hash(suricata_data.get('alert', {}).get('category', '')) % 1000,
            
            # Flow information
            'flow_bytes_toclient': suricata_data.get('flow', {}).get('bytes_toclient', 0),
            'flow_bytes_toserver': suricata_data.get('flow', {}).get('bytes_toserver', 0),
            'flow_pkts_toclient': suricata_data.get('flow', {}).get('pkts_toclient', 0),
            'flow_pkts_toserver': suricata_data.get('flow', {}).get('pkts_toserver', 0),
            
            # Protocol analysis
            'proto': hash(suricata_data.get('proto', '')) % 100,
            'app_proto': hash(suricata_data.get('app_proto', '')) % 100,
        }
        
        return features


class HybridMLPipeline:
    """Pipeline híbrido de Machine Learning para detecção de DDoS"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_extractor = FeatureExtractor()
        self.reputation_threshold = 0.7
        self.anomaly_threshold = -0.5
        
        # Inicializar modelos
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa os modelos de ML com carregamento automático inteligente"""
        
        # Tentar carregar modelos em ordem de prioridade
        model_loaded = self._try_load_optimized_models()
        
        if not model_loaded:
            model_loaded = self._try_load_hybrid_advanced_models()
        
        if not model_loaded:
            model_loaded = self._try_load_simple_model()
        
        if not model_loaded:
            self._initialize_default_models()
            logger.info("🔧 Usando modelos padrão (não treinados)")
        
        logger.info("✅ Modelos ML inicializados")

    def _try_load_optimized_models(self) -> bool:
        """Tenta carregar modelos otimizados (1ª prioridade)"""
        try:
            models_path = "/app/models/hybrid_advanced"
            
            # Verificar se existem modelos otimizados
            xgb_optimized = f"{models_path}/xgboost_optimized_simple.pkl"
            if_optimized = f"{models_path}/isolation_forest_optimized.pkl"
            scaler_path = f"{models_path}/scalers_advanced.pkl"
            
            if os.path.exists(xgb_optimized) and os.path.exists(if_optimized):
                # Carregar XGBoost otimizado
                with open(xgb_optimized, 'rb') as f:
                    self.models['supervised_xgb'] = pickle.load(f)
                
                # Carregar Isolation Forest otimizado
                with open(if_optimized, 'rb') as f:
                    self.models['unsupervised_if'] = pickle.load(f)
                
                # Random Forest padrão (não há versão otimizada)
                self.models['supervised_rf'] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Carregar scaler se existir
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                else:
                    self.scaler = StandardScaler()
                
                logger.info("🚀 Modelos OTIMIZADOS carregados com sucesso", 
                           xgb=xgb_optimized, isolation_forest=if_optimized)
                return True
                
        except Exception as e:
            logger.warning("❌ Falha ao carregar modelos otimizados", error=str(e))
        
        return False

    def _try_load_hybrid_advanced_models(self) -> bool:
        """Tenta carregar modelos híbridos avançados (2ª prioridade)"""
        try:
            models_path = "/app/models/hybrid_advanced"
            hybrid_models_file = f"{models_path}/hybrid_advanced_models.pkl"
            
            if os.path.exists(hybrid_models_file):
                with open(hybrid_models_file, 'rb') as f:
                    models_data = pickle.load(f)
                
                # Mapear modelos avançados para estrutura esperada
                if 'general_xgboost' in models_data:
                    self.models['supervised_xgb'] = models_data['general_xgboost']
                if 'general_isolation_forest' in models_data:
                    self.models['unsupervised_if'] = models_data['general_isolation_forest']
                if 'ddos_random_forest' in models_data:
                    self.models['supervised_rf'] = models_data['ddos_random_forest']
                elif 'ddos_xgboost' in models_data:
                    # Se não há RF, usar XGBoost especialista como RF
                    self.models['supervised_rf'] = models_data['ddos_xgboost']
                
                # Carregar scaler
                if 'ddos_scaler' in models_data:
                    self.scaler = models_data['ddos_scaler']
                else:
                    self.scaler = StandardScaler()
                
                logger.info("🎯 Modelos HÍBRIDOS AVANÇADOS carregados com sucesso", 
                           path=hybrid_models_file)
                return True
                
        except Exception as e:
            logger.warning("❌ Falha ao carregar modelos híbridos avançados", error=str(e))
        
        return False

    def _try_load_simple_model(self) -> bool:
        """Tenta carregar modelo simples (3ª prioridade)"""
        try:
            simple_model_path = "/app/models/simple/rf_model.pkl"
            
            if os.path.exists(simple_model_path):
                with open(simple_model_path, 'rb') as f:
                    self.models['supervised_rf'] = pickle.load(f)
                
                # Usar modelos padrão para os outros
                self.models['supervised_xgb'] = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                
                self.models['unsupervised_if'] = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                
                self.scaler = StandardScaler()
                
                logger.info("📊 Modelo SIMPLES carregado com sucesso", 
                           path=simple_model_path)
                return True
                
        except Exception as e:
            logger.warning("❌ Falha ao carregar modelo simples", error=str(e))
        
        return False

    def _initialize_default_models(self):
        """Inicializa modelos padrão não treinados (4ª prioridade)"""
        # Modelo supervisionado - Random Forest para ataques conhecidos
        self.models['supervised_rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Modelo supervisionado - XGBoost para ataques conhecidos
        self.models['supervised_xgb'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # Modelo não-supervisionado - Isolation Forest para anomalias
        self.models['unsupervised_if'] = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # Scaler padrão
        self.scaler = StandardScaler()
    
    def load_models(self, model_path: str):
        """Carrega modelos pré-treinados"""
        try:
            with open(f"{model_path}/supervised_rf.pkl", 'rb') as f:
                self.models['supervised_rf'] = pickle.load(f)
            
            with open(f"{model_path}/supervised_xgb.pkl", 'rb') as f:
                self.models['supervised_xgb'] = pickle.load(f)
            
            with open(f"{model_path}/unsupervised_if.pkl", 'rb') as f:
                self.models['unsupervised_if'] = pickle.load(f)
            
            with open(f"{model_path}/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info("Modelos carregados com sucesso", model_path=model_path)
        except Exception as e:
            logger.warning("Falha ao carregar modelos", error=str(e))
    
    def save_models(self, model_path: str):
        """Salva modelos treinados"""
        try:
            with open(f"{model_path}/supervised_rf.pkl", 'wb') as f:
                pickle.dump(self.models['supervised_rf'], f)
            
            with open(f"{model_path}/supervised_xgb.pkl", 'wb') as f:
                pickle.dump(self.models['supervised_xgb'], f)
            
            with open(f"{model_path}/unsupervised_if.pkl", 'wb') as f:
                pickle.dump(self.models['unsupervised_if'], f)
            
            with open(f"{model_path}/scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info("Modelos salvos com sucesso", model_path=model_path)
        except Exception as e:
            logger.error("Falha ao salvar modelos", error=str(e))
    
    def preprocess_data(self, raw_data: Dict) -> np.ndarray:
        """Pré-processa dados para os modelos ML"""
        # Determinar tipo de dados (Suricata ou NetFlow)
        if 'alert' in raw_data:
            features = self.feature_extractor.extract_suricata_features(raw_data)
        else:
            features = self.feature_extractor.extract_flow_features(raw_data)
        
        # Converter para array numpy
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Normalizar características
        try:
            feature_vector = self.scaler.transform(feature_vector)
        except:
            # Se o scaler não foi treinado, usar fit_transform
            feature_vector = self.scaler.fit_transform(feature_vector)
        
        return feature_vector
    
    def detect_supervised(self, features: np.ndarray) -> Tuple[bool, float, str]:
        """Detecção supervisionada de ataques conhecidos"""
        # Predição com Random Forest
        rf_pred = self.models['supervised_rf'].predict_proba(features)[0]
        rf_threat_score = rf_pred[1] if len(rf_pred) > 1 else 0
        
        # Predição com XGBoost
        xgb_pred = self.models['supervised_xgb'].predict_proba(features)[0]
        xgb_threat_score = xgb_pred[1] if len(xgb_pred) > 1 else 0
        
        # Combinar scores (ensemble)
        combined_score = (rf_threat_score + xgb_threat_score) / 2
        
        is_attack = combined_score > self.reputation_threshold
        attack_type = "known_ddos" if is_attack else "benign"
        
        return is_attack, combined_score, attack_type
    
    def detect_unsupervised(self, features: np.ndarray) -> Tuple[bool, float, str]:
        """Detecção não-supervisionada de anomalias"""
        # Isolation Forest para detecção de anomalias
        anomaly_score = self.models['unsupervised_if'].decision_function(features)[0]
        is_anomaly = anomaly_score < self.anomaly_threshold
        
        # Normalizar score para 0-1
        normalized_score = max(0, min(1, (self.anomaly_threshold - anomaly_score) / 2))
        
        attack_type = "zero_day" if is_anomaly else "benign"
        
        return is_anomaly, normalized_score, attack_type
    
    def analyze_traffic(self, raw_data: Dict) -> Optional[ThreatScore]:
        """Análise principal do tráfego"""
        try:
            # Pré-processar dados
            features = self.preprocess_data(raw_data)
            
            # Detecção supervisionada
            supervised_threat, supervised_score, supervised_type = self.detect_supervised(features)
            
            # Detecção não-supervisionada
            unsupervised_threat, unsupervised_score, unsupervised_type = self.detect_unsupervised(features)
            
            # Combinar resultados
            if supervised_threat:
                threat_score = ThreatScore(
                    ip=raw_data.get('src_ip', ''),
                    score=supervised_score,
                    attack_type=supervised_type,
                    confidence=supervised_score,
                    detection_method="supervised",
                    timestamp=datetime.now()
                )
            elif unsupervised_threat:
                threat_score = ThreatScore(
                    ip=raw_data.get('src_ip', ''),
                    score=unsupervised_score,
                    attack_type=unsupervised_type,
                    confidence=unsupervised_score,
                    detection_method="unsupervised",
                    timestamp=datetime.now()
                )
            else:
                return None
            
            logger.info("Ameaça detectada", 
                       ip=threat_score.ip,
                       score=threat_score.score,
                       attack_type=threat_score.attack_type,
                       method=threat_score.detection_method)
            
            return threat_score
            
        except Exception as e:
            logger.error("Erro na análise de tráfego", error=str(e))
            return None


class ReputationSystem:
    """Sistema de reputação para IPs maliciosos"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.reputation_key_prefix = "reputation:"
        self.validation_threshold = 3  # Número mínimo de validações
        self.decay_rate = 0.1  # Taxa de decaimento da reputação
    
    def update_reputation(self, threat_score: ThreatScore) -> float:
        """Atualiza a reputação de um IP"""
        key = f"{self.reputation_key_prefix}{threat_score.ip}"
        
        # Obter reputação atual
        current_data = self.redis_client.get(key)
        if current_data:
            reputation_data = json.loads(current_data)
            current_score = reputation_data.get('score', 0)
            validation_count = reputation_data.get('validations', 0)
        else:
            current_score = 0
            validation_count = 0
        
        # Calcular nova reputação
        new_score = min(1.0, current_score + threat_score.score * 0.3)
        validation_count += 1
        
        # Salvar dados atualizados
        reputation_data = {
            'score': new_score,
            'validations': validation_count,
            'last_seen': threat_score.timestamp.isoformat(),
            'attack_types': [threat_score.attack_type],
            'detection_methods': [threat_score.detection_method]
        }
        
        # TTL baseado na reputação (maior reputação = maior TTL)
        ttl = int(3600 * (1 + new_score))
        self.redis_client.setex(key, ttl, json.dumps(reputation_data))
        
        logger.info("Reputação atualizada",
                   ip=threat_score.ip,
                   new_score=new_score,
                   validations=validation_count)
        
        return new_score
    
    def should_block_ip(self, ip: str) -> Tuple[bool, Dict]:
        """Determina se um IP deve ser bloqueado"""
        key = f"{self.reputation_key_prefix}{ip}"
        current_data = self.redis_client.get(key)
        
        if not current_data:
            return False, {}
        
        reputation_data = json.loads(current_data)
        score = reputation_data.get('score', 0)
        validations = reputation_data.get('validations', 0)
        
        # Critérios para bloqueio
        should_block = (
            score > 0.7 and validations >= self.validation_threshold
        ) or score > 0.9
        
        return should_block, reputation_data
    
    def decay_reputations(self):
        """Aplica decaimento às reputações (executar periodicamente)"""
        # Esta função seria executada em uma task separada
        pass


class MLProcessor:
    """Processador principal de ML"""
    
    def __init__(self, kafka_config: Dict, redis_config: Dict):
        self.kafka_config = kafka_config
        self.redis_client = redis.Redis(**redis_config)
        
        # Inicializar componentes
        self.ml_pipeline = HybridMLPipeline(self.redis_client)
        self.reputation_system = ReputationSystem(self.redis_client)
        
        # Configurar Kafka
        self.consumer = KafkaConsumer(
            'traffic-logs',
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='ml-processor'
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        logger.info("ML Processor inicializado")
    
    async def process_messages(self):
        """Loop principal de processamento de mensagens"""
        logger.info("Iniciando processamento de mensagens")
        
        for message in self.consumer:
            try:
                raw_data = message.value
                
                # Análise de tráfego
                threat_score = self.ml_pipeline.analyze_traffic(raw_data)
                
                if threat_score:
                    # Atualizar sistema de reputação
                    reputation_score = self.reputation_system.update_reputation(threat_score)
                    
                    # Verificar se deve bloquear IP
                    should_block, reputation_data = self.reputation_system.should_block_ip(threat_score.ip)
                    
                    if should_block:
                        # Publicar IP malicioso para BGP Controller
                        malicious_ip_data = {
                            'ip': threat_score.ip,
                            'threat_score': threat_score.score,
                            'reputation_score': reputation_score,
                            'attack_type': threat_score.attack_type,
                            'confidence': threat_score.confidence,
                            'detection_method': threat_score.detection_method,
                            'timestamp': threat_score.timestamp.isoformat(),
                            'ttl': threat_score.ttl,
                            'validations': reputation_data.get('validations', 0)
                        }
                        
                        self.producer.send('malicious-ips', malicious_ip_data)
                        
                        logger.info("IP malicioso publicado",
                                   ip=threat_score.ip,
                                   attack_type=threat_score.attack_type,
                                   score=threat_score.score)
                
            except Exception as e:
                logger.error("Erro no processamento de mensagem", error=str(e))
    
    def train_models(self, training_data_path: str):
        """Treina os modelos com dados históricos"""
        # Implementar treinamento com datasets como CICDDoS2019
        logger.info("Iniciando treinamento de modelos")
        # TODO: Implementar treinamento completo
        pass


if __name__ == "__main__":
    # Configuração
    kafka_config = {
        'bootstrap_servers': ['kafka:29092']
    }
    
    redis_config = {
        'host': 'redis',
        'port': 6379,
        'db': 0
    }
    
    # Inicializar processador
    processor = MLProcessor(kafka_config, redis_config)
    
    # Executar processamento
    asyncio.run(processor.process_messages())
