"""
Data Ingestion - Módulo de Ingestão de Dados

Este módulo implementa a ingestão de dados de múltiplas fontes:
1. Suricata logs (eve.json)
2. NetFlow/sFlow collectors
3. Normalização e envio para Kafka
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator
from pathlib import Path
import aiofiles
from kafka import KafkaProducer
import structlog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuração de logging
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


class SuricataLogHandler(FileSystemEventHandler):
    """Handler para monitorar logs do Suricata"""
    
    def __init__(self, kafka_producer: KafkaProducer, log_file_path: str):
        self.kafka_producer = kafka_producer
        self.log_file_path = log_file_path
        self.file_position = 0
        
        # Se o arquivo existe, posicionar no final
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                f.seek(0, 2)  # Ir para o final do arquivo
                self.file_position = f.tell()
    
    def on_modified(self, event):
        """Chamado quando o arquivo de log é modificado"""
        if event.src_path == self.log_file_path and not event.is_directory:
            self.process_new_logs()
    
    def process_new_logs(self):
        """Processa novas linhas do log do Suricata"""
        try:
            with open(self.log_file_path, 'r') as f:
                f.seek(self.file_position)
                
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            log_entry = json.loads(line)
                            self.process_suricata_entry(log_entry)
                        except json.JSONDecodeError:
                            logger.warning("Linha JSON inválida no log Suricata", line=line)
                
                # Atualizar posição
                self.file_position = f.tell()
                
        except Exception as e:
            logger.error("Erro ao processar logs Suricata", error=str(e))
    
    def process_suricata_entry(self, log_entry: Dict):
        """Processa entrada do log Suricata"""
        try:
            # Normalizar dados do Suricata
            normalized_data = self.normalize_suricata_data(log_entry)
            
            if normalized_data:
                # Enviar para Kafka
                self.kafka_producer.send('traffic-logs', normalized_data)
                
                logger.debug("Log Suricata processado",
                           event_type=log_entry.get('event_type'),
                           src_ip=normalized_data.get('src_ip'),
                           dst_ip=normalized_data.get('dst_ip'))
                
        except Exception as e:
            logger.error("Erro ao processar entrada Suricata", error=str(e))
    
    def normalize_suricata_data(self, log_entry: Dict) -> Optional[Dict]:
        """Normaliza dados do Suricata para formato comum"""
        try:
            # Filtrar apenas eventos relevantes
            event_type = log_entry.get('event_type', '')
            if event_type not in ['alert', 'flow', 'netflow']:
                return None
            
            # Extrair informações básicas
            timestamp = log_entry.get('timestamp', datetime.now().isoformat())
            src_ip = log_entry.get('src_ip', '')
            dest_ip = log_entry.get('dest_ip', '')
            src_port = log_entry.get('src_port', 0)
            dest_port = log_entry.get('dest_port', 0)
            proto = log_entry.get('proto', '')
            
            # Dados normalizados
            normalized = {
                'timestamp': timestamp,
                'source': 'suricata',
                'event_type': event_type,
                'src_ip': src_ip,
                'dst_ip': dest_ip,
                'src_port': src_port,
                'dst_port': dest_port,
                'protocol': proto,
                'raw_data': log_entry
            }
            
            # Adicionar informações específicas baseadas no tipo de evento
            if event_type == 'alert':
                alert = log_entry.get('alert', {})
                normalized.update({
                    'alert_signature': alert.get('signature', ''),
                    'alert_category': alert.get('category', ''),
                    'alert_severity': alert.get('severity', 0),
                    'alert_signature_id': alert.get('signature_id', 0)
                })
            
            elif event_type == 'flow':
                flow = log_entry.get('flow', {})
                normalized.update({
                    'bytes_toserver': flow.get('bytes_toserver', 0),
                    'bytes_toclient': flow.get('bytes_toclient', 0),
                    'pkts_toserver': flow.get('pkts_toserver', 0),
                    'pkts_toclient': flow.get('pkts_toclient', 0),
                    'flow_age': flow.get('age', 0),
                    'flow_state': flow.get('state', '')
                })
            
            return normalized
            
        except Exception as e:
            logger.error("Erro na normalização Suricata", error=str(e))
            return None


class NetFlowCollector:
    """Coletor de dados NetFlow/sFlow"""
    
    def __init__(self, kafka_producer: KafkaProducer, bind_host: str = '0.0.0.0', bind_port: int = 2055):
        self.kafka_producer = kafka_producer
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.running = False
        
        # Cache para agregação de fluxos
        self.flow_cache = {}
        self.cache_timeout = 300  # 5 minutos
    
    async def start_collector(self):
        """Inicia o coletor NetFlow"""
        logger.info("Iniciando coletor NetFlow", host=self.bind_host, port=self.bind_port)
        
        # Criar socket UDP para receber NetFlow
        import socket
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.bind_host, self.bind_port))
        sock.settimeout(1.0)  # Timeout de 1 segundo
        
        self.running = True
        
        while self.running:
            try:
                data, addr = sock.recvfrom(1500)  # MTU padrão
                await self.process_netflow_packet(data, addr)
                
            except socket.timeout:
                continue
            except Exception as e:
                logger.error("Erro ao receber pacote NetFlow", error=str(e))
        
        sock.close()
    
    async def process_netflow_packet(self, data: bytes, addr: tuple):
        """Processa pacote NetFlow recebido"""
        try:
            # Parse simples do NetFlow v5 (header básico)
            if len(data) < 24:  # Tamanho mínimo do header NetFlow v5
                return
            
            # Extrair header NetFlow v5
            import struct
            header = struct.unpack('!HHIIIIBBH', data[:24])
            
            version = header[0]
            count = header[1]
            sys_uptime = header[2]
            unix_secs = header[3]
            unix_nsecs = header[4]
            flow_sequence = header[5]
            engine_type = header[6]
            engine_id = header[7]
            sampling_interval = header[8]
            
            if version != 5:
                logger.debug("Versão NetFlow não suportada", version=version)
                return
            
            # Processar registros de fluxo
            record_offset = 24
            for i in range(count):
                if record_offset + 48 <= len(data):  # Tamanho do registro NetFlow v5
                    record = struct.unpack('!IIIIHHIIIIHHBBBBHHBBH', data[record_offset:record_offset + 48])
                    
                    flow_record = {
                        'src_ip': self.int_to_ip(record[0]),
                        'dst_ip': self.int_to_ip(record[1]),
                        'next_hop': self.int_to_ip(record[2]),
                        'input_snmp': record[3],
                        'output_snmp': record[4],
                        'packets': record[5],
                        'bytes': record[6],
                        'first_switched': record[7],
                        'last_switched': record[8],
                        'src_port': record[9],
                        'dst_port': record[10],
                        'tcp_flags': record[12],
                        'protocol': record[13],
                        'tos': record[14],
                        'src_as': record[15],
                        'dst_as': record[16],
                        'src_mask': record[17],
                        'dst_mask': record[18]
                    }
                    
                    # Normalizar e enviar para Kafka
                    normalized_flow = self.normalize_netflow_data(flow_record, unix_secs)
                    
                    if normalized_flow:
                        self.kafka_producer.send('traffic-logs', normalized_flow)
                        
                        logger.debug("NetFlow processado",
                                   src_ip=flow_record['src_ip'],
                                   dst_ip=flow_record['dst_ip'],
                                   packets=flow_record['packets'],
                                   bytes=flow_record['bytes'])
                    
                    record_offset += 48
                
        except Exception as e:
            logger.error("Erro ao processar pacote NetFlow", error=str(e))
    
    def int_to_ip(self, ip_int: int) -> str:
        """Converte inteiro para string IP"""
        return f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"
    
    def normalize_netflow_data(self, flow_record: Dict, timestamp: int) -> Optional[Dict]:
        """Normaliza dados NetFlow para formato comum"""
        try:
            # Calcular duração do fluxo
            duration = max(1, (flow_record['last_switched'] - flow_record['first_switched']) / 1000)
            
            normalized = {
                'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                'source': 'netflow',
                'event_type': 'flow',
                'src_ip': flow_record['src_ip'],
                'dst_ip': flow_record['dst_ip'],
                'src_port': flow_record['src_port'],
                'dst_port': flow_record['dst_port'],
                'protocol': flow_record['protocol'],
                'packets': flow_record['packets'],
                'bytes': flow_record['bytes'],
                'duration': duration,
                'tcp_flags': flow_record['tcp_flags'],
                'next_hop': flow_record['next_hop'],
                'src_as': flow_record['src_as'],
                'dst_as': flow_record['dst_as'],
                'raw_data': flow_record
            }
            
            return normalized
            
        except Exception as e:
            logger.error("Erro na normalização NetFlow", error=str(e))
            return None
    
    def stop_collector(self):
        """Para o coletor NetFlow"""
        self.running = False


class DataIngestionService:
    """Serviço principal de ingestão de dados"""
    
    def __init__(self, kafka_config: Dict, ingestion_config: Dict):
        self.kafka_config = kafka_config
        self.ingestion_config = ingestion_config
        
        # Inicializar Kafka Producer
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            compression_type='gzip',
            batch_size=16384,
            linger_ms=10
        )
        
        # Componentes de ingestão
        self.suricata_handler = None
        self.netflow_collector = None
        self.file_observer = None
        
        logger.info("Serviço de ingestão inicializado")
    
    async def start_suricata_ingestion(self):
        """Inicia ingestão de logs Suricata"""
        suricata_log_path = self.ingestion_config.get('suricata_log_path', '/var/log/suricata/eve.json')
        
        if not os.path.exists(suricata_log_path):
            logger.warning("Arquivo de log Suricata não encontrado", path=suricata_log_path)
            return
        
        # Criar handler para o arquivo de log
        self.suricata_handler = SuricataLogHandler(self.kafka_producer, suricata_log_path)
        
        # Configurar observer para monitorar mudanças no arquivo
        self.file_observer = Observer()
        self.file_observer.schedule(
            self.suricata_handler,
            path=os.path.dirname(suricata_log_path),
            recursive=False
        )
        
        self.file_observer.start()
        
        # Processar logs existentes
        self.suricata_handler.process_new_logs()
        
        logger.info("Ingestão Suricata iniciada", log_path=suricata_log_path)
    
    async def start_netflow_ingestion(self):
        """Inicia ingestão NetFlow"""
        netflow_host = self.ingestion_config.get('netflow_host', '0.0.0.0')
        netflow_port = self.ingestion_config.get('netflow_port', 2055)
        
        self.netflow_collector = NetFlowCollector(
            self.kafka_producer,
            netflow_host,
            netflow_port
        )
        
        # Iniciar coletor em task separada
        asyncio.create_task(self.netflow_collector.start_collector())
        
        logger.info("Ingestão NetFlow iniciada", host=netflow_host, port=netflow_port)
    
    async def start_ingestion(self):
        """Inicia todos os serviços de ingestão"""
        logger.info("Iniciando serviços de ingestão de dados")
        
        # Iniciar ingestão Suricata
        if self.ingestion_config.get('enable_suricata', True):
            await self.start_suricata_ingestion()
        
        # Iniciar ingestão NetFlow
        if self.ingestion_config.get('enable_netflow', True):
            await self.start_netflow_ingestion()
        
        # Manter serviço rodando
        try:
            while True:
                await asyncio.sleep(10)
                # Aqui poderia adicionar health checks
                
        except KeyboardInterrupt:
            logger.info("Parando serviços de ingestão")
            self.stop_ingestion()
    
    def stop_ingestion(self):
        """Para todos os serviços de ingestão"""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
        
        if self.netflow_collector:
            self.netflow_collector.stop_collector()
        
        if self.kafka_producer:
            self.kafka_producer.close()
        
        logger.info("Serviços de ingestão parados")


if __name__ == "__main__":
    # Configuração
    kafka_config = {
        'bootstrap_servers': ['kafka:29092']
    }
    
    ingestion_config = {
        'enable_suricata': True,
        'enable_netflow': True,
        'suricata_log_path': '/var/log/suricata/eve.json',
        'netflow_host': '0.0.0.0',
        'netflow_port': 2055
    }
    
    # Inicializar serviço
    service = DataIngestionService(kafka_config, ingestion_config)
    
    # Executar ingestão
    asyncio.run(service.start_ingestion())
