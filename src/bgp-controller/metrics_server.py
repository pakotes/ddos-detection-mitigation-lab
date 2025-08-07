#!/usr/bin/env python3
"""
Servidor de métricas básico para BGP Controller
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import threading
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Métricas do BGP Controller
BGP_ROUTES_ANNOUNCED = Counter('bgp_routes_announced_total', 'Total de rotas BGP anunciadas')
BGP_ROUTES_WITHDRAWN = Counter('bgp_routes_withdrawn_total', 'Total de rotas BGP retiradas')
BGP_ACTIVE_ROUTES = Gauge('bgp_active_routes', 'Rotas BGP ativas')
BGP_PROCESSING_TIME = Histogram('bgp_processing_time_seconds', 'Tempo de processamento BGP')

class BGPMetricsServer:
    """Servidor de métricas para BGP Controller"""
    
    def __init__(self, port=8001):
        self.port = port
        self.running = False
        
    def start(self):
        """Inicia o servidor de métricas"""
        try:
            start_http_server(self.port)
            self.running = True
            logger.info(f"Servidor de métricas BGP iniciado na porta {self.port}")
            
            # Iniciar thread para gerar métricas dummy
            metrics_thread = threading.Thread(target=self._generate_dummy_metrics)
            metrics_thread.daemon = True
            metrics_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Erro ao iniciar servidor de métricas BGP: {e}")
            return False
    
    def _generate_dummy_metrics(self):
        """Gera métricas dummy para demonstração"""
        import random
        
        active_routes = 0
        
        while self.running:
            try:
                # Simular atividade BGP
                if random.random() < 0.3:  # 30% chance de anunciar rota
                    BGP_ROUTES_ANNOUNCED.inc()
                    active_routes += 1
                    BGP_PROCESSING_TIME.observe(random.uniform(0.01, 0.5))
                
                if random.random() < 0.1 and active_routes > 0:  # 10% chance de retirar rota
                    BGP_ROUTES_WITHDRAWN.inc()
                    active_routes -= 1
                    BGP_PROCESSING_TIME.observe(random.uniform(0.01, 0.3))
                
                BGP_ACTIVE_ROUTES.set(max(0, active_routes))
                
                time.sleep(10)  # Atualizar a cada 10 segundos
                
            except Exception as e:
                logger.error(f"Erro ao gerar métricas BGP: {e}")
                time.sleep(15)

if __name__ == "__main__":
    server = BGPMetricsServer(8001)
    
    if server.start():
        logger.info("Servidor de métricas BGP Controller rodando...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Parando servidor de métricas BGP...")
            server.running = False
    else:
        logger.error("Falha ao iniciar servidor de métricas BGP")
