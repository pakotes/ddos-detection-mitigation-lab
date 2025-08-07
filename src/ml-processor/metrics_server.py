#!/usr/bin/env python3
"""
Servidor de métricas básico para expor métricas Prometheus
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import threading
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Métricas básicas
PROCESSED_TRAFFIC_TOTAL = Counter('ddos_processed_traffic_total', 'Total de tráfego processado')
DETECTION_LATENCY = Histogram('ddos_detection_latency_seconds', 'Latência de detecção em segundos')
ACTIVE_CONNECTIONS = Gauge('ddos_active_connections', 'Conexões ativas')
MALICIOUS_IPS_DETECTED = Counter('ddos_malicious_ips_total', 'Total de IPs maliciosos detectados')
ML_MODEL_ACCURACY = Gauge('ddos_ml_model_accuracy', 'Precisão do modelo ML')

class MetricsServer:
    """Servidor de métricas básico"""
    
    def __init__(self, port=8000):
        self.port = port
        self.running = False
        
    def start(self):
        """Inicia o servidor de métricas"""
        try:
            start_http_server(self.port)
            self.running = True
            logger.info(f"Servidor de métricas iniciado na porta {self.port}")
            
            # Iniciar thread para gerar métricas dummy
            metrics_thread = threading.Thread(target=self._generate_dummy_metrics)
            metrics_thread.daemon = True
            metrics_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Erro ao iniciar servidor de métricas: {e}")
            return False
    
    def _generate_dummy_metrics(self):
        """Gera métricas dummy para demonstração"""
        import random
        
        while self.running:
            try:
                # Simular métricas básicas
                PROCESSED_TRAFFIC_TOTAL.inc(random.randint(1, 10))
                DETECTION_LATENCY.observe(random.uniform(0.001, 0.1))
                ACTIVE_CONNECTIONS.set(random.randint(50, 200))
                
                if random.random() < 0.1:  # 10% chance de detectar IP malicioso
                    MALICIOUS_IPS_DETECTED.inc()
                
                ML_MODEL_ACCURACY.set(random.uniform(0.85, 0.98))
                
                time.sleep(5)  # Atualizar a cada 5 segundos
                
            except Exception as e:
                logger.error(f"Erro ao gerar métricas: {e}")
                time.sleep(10)

if __name__ == "__main__":
    server = MetricsServer(8000)
    
    if server.start():
        logger.info("Servidor de métricas ML Processor rodando...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Parando servidor de métricas...")
            server.running = False
    else:
        logger.error("Falha ao iniciar servidor de métricas")
