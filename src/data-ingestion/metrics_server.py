#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servidor de metricas basico para Data Ingestion
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import threading
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metricas de Data Ingestion
INGESTED_PACKETS_TOTAL = Counter('data_ingestion_packets_total', 'Total de pacotes ingeridos')
INGESTION_LATENCY = Histogram('data_ingestion_latency_seconds', 'Latencia de ingestao')
SURICATA_EVENTS = Counter('data_ingestion_suricata_events_total', 'Eventos Suricata processados')
NETFLOW_RECORDS = Counter('data_ingestion_netflow_records_total', 'Registros NetFlow processados')

class DataIngestionMetricsServer:
    """Servidor de metricas para Data Ingestion"""
    
    def __init__(self, port=8002):
        self.port = port
        self.running = False
        
    def start(self):
        """Inicia o servidor de métricas"""
        try:
            start_http_server(self.port)
            self.running = True
            logger.info(f"Servidor de métricas Data Ingestion iniciado na porta {self.port}")
            
            # Iniciar thread para gerar métricas dummy
            metrics_thread = threading.Thread(target=self._generate_dummy_metrics)
            metrics_thread.daemon = True
            metrics_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Erro ao iniciar servidor de métricas Data Ingestion: {e}")
            return False
    
    def _generate_dummy_metrics(self):
        """Gera métricas dummy para demonstração"""
        import random
        
        while self.running:
            try:
                # Simular ingestão de dados
                packets = random.randint(50, 200)
                INGESTED_PACKETS_TOTAL.inc(packets)
                INGESTION_LATENCY.observe(random.uniform(0.001, 0.05))
                
                # Eventos Suricata
                if random.random() < 0.8:  # 80% chance de eventos Suricata
                    SURICATA_EVENTS.inc(random.randint(1, 20))
                
                # Registros NetFlow
                if random.random() < 0.6:  # 60% chance de NetFlow
                    NETFLOW_RECORDS.inc(random.randint(10, 50))
                
                time.sleep(3)  # Atualizar a cada 3 segundos
                
            except Exception as e:
                logger.error(f"Erro ao gerar métricas Data Ingestion: {e}")
                time.sleep(10)

if __name__ == "__main__":
    server = DataIngestionMetricsServer(8002)
    
    if server.start():
        logger.info("Servidor de métricas Data Ingestion rodando...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Parando servidor de métricas Data Ingestion...")
            server.running = False
    else:
        logger.error("Falha ao iniciar servidor de métricas Data Ingestion")
