"""
BGP Flow Controller - Controlador BGP para Mitigação Automática de DDoS

Este módulo implementa o controlador BGP que:
1. Consome IPs maliciosos do Kafka (malicious-ips)
2. Decide estratégias de mitigação (Flowspec vs Blackhole)
3. Gera anúncios BGP via ExaBGP
4. Gerencia TTL automático para desanúncios
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from kafka import KafkaConsumer, KafkaProducer
import structlog

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


@dataclass
class BGPRoute:
    """Classe para representar uma rota BGP"""
    ip: str
    prefix_length: int
    next_hop: str
    community: str
    action: str  # 'announce' or 'withdraw'
    route_type: str  # 'blackhole' or 'flowspec'
    timestamp: datetime
    ttl: int
    attack_type: str
    threat_score: float


class BGPFlowspecGenerator:
    """Gerador de regras BGP Flowspec"""
    
    def __init__(self):
        self.flowspec_communities = {
            'rate_limit_1mbps': '65000:1',
            'rate_limit_10mbps': '65000:10', 
            'rate_limit_100mbps': '65000:100',
            'redirect_vrf': '65000:9999',
            'discard': '65000:666'
        }
    
    def generate_flowspec_rule(self, malicious_ip: Dict) -> str:
        """Gera regra Flowspec baseada no tipo de ataque"""
        ip = malicious_ip['ip']
        attack_type = malicious_ip.get('attack_type', 'unknown')
        threat_score = malicious_ip.get('threat_score', 0.5)
        
        # Determinar ação baseada no tipo de ataque e score
        if attack_type in ['volumetric', 'amplification'] or threat_score > 0.9:
            action = 'discard'
            community = self.flowspec_communities['discard']
        elif attack_type in ['application_layer'] or threat_score > 0.7:
            action = 'rate_limit_1mbps'
            community = self.flowspec_communities['rate_limit_1mbps']
        else:
            action = 'rate_limit_10mbps'
            community = self.flowspec_communities['rate_limit_10mbps']
        
        # Gerar regra Flowspec
        flowspec_rule = f"""
flow route {{
    match {{
        source {ip}/32;
    }}
    then {{
        {action};
        community {community};
    }}
}}
"""
        
        logger.info("Regra Flowspec gerada",
                   ip=ip,
                   action=action,
                   attack_type=attack_type,
                   threat_score=threat_score)
        
        return flowspec_rule.strip()


class BGPBlackholeGenerator:
    """Gerador de rotas Blackhole BGP"""
    
    def __init__(self, blackhole_community: str = "65000:666"):
        self.blackhole_community = blackhole_community
        self.blackhole_next_hop = "192.0.2.1"  # RFC 5735 - Test-Net
    
    def generate_blackhole_route(self, malicious_ip: Dict) -> str:
        """Gera rota blackhole para IP malicioso"""
        ip = malicious_ip['ip']
        
        # Gerar anúncio blackhole
        blackhole_route = f"announce route {ip}/32 next-hop {self.blackhole_next_hop} community [{self.blackhole_community}]"
        
        logger.info("Rota blackhole gerada",
                   ip=ip,
                   next_hop=self.blackhole_next_hop,
                   community=self.blackhole_community)
        
        return blackhole_route


class ExaBGPController:
    """Controlador do ExaBGP"""
    
    def __init__(self, exabgp_config_path: str = "/etc/exabgp/exabgp.conf"):
        self.config_path = exabgp_config_path
        self.process = None
        self.pipe_path = "/var/run/exabgp/exabgp.in"
        
        # Criar diretório para pipe se não existir
        os.makedirs(os.path.dirname(self.pipe_path), exist_ok=True)
    
    def start_exabgp(self):
        """Inicia o processo ExaBGP"""
        try:
            cmd = ["exabgp", self.config_path]
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info("ExaBGP iniciado", pid=self.process.pid)
            return True
        except Exception as e:
            logger.error("Falha ao iniciar ExaBGP", error=str(e))
            return False
    
    def send_bgp_command(self, command: str):
        """Envia comando BGP para ExaBGP"""
        try:
            if self.process and self.process.poll() is None:
                self.process.stdin.write(command + '\n')
                self.process.stdin.flush()
                logger.info("Comando BGP enviado", command=command)
            else:
                logger.error("ExaBGP não está rodando")
        except Exception as e:
            logger.error("Falha ao enviar comando BGP", error=str(e))
    
    def stop_exabgp(self):
        """Para o processo ExaBGP"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            logger.info("ExaBGP parado")


class MitigationStrategy:
    """Estratégia de mitigação para diferentes tipos de ataques"""
    
    def __init__(self):
        self.strategies = {
            'volumetric': 'blackhole',
            'amplification': 'blackhole',
            'application_layer': 'flowspec',
            'protocol': 'flowspec',
            'zero_day': 'flowspec',
            'known_ddos': 'flowspec'
        }
    
    def get_strategy(self, attack_type: str, threat_score: float) -> str:
        """Determina estratégia de mitigação"""
        # Strategy override baseado em threat score alto
        if threat_score > 0.95:
            return 'blackhole'
        
        return self.strategies.get(attack_type, 'flowspec')


class TTLManager:
    """Gerenciador de TTL para anúncios BGP"""
    
    def __init__(self):
        self.active_routes = {}  # ip -> BGPRoute
    
    def add_route(self, route: BGPRoute):
        """Adiciona rota ativa com TTL"""
        self.active_routes[route.ip] = route
        logger.info("Rota adicionada ao TTL manager",
                   ip=route.ip,
                   ttl=route.ttl,
                   route_type=route.route_type)
    
    def get_expired_routes(self) -> List[BGPRoute]:
        """Retorna rotas que expiraram"""
        now = datetime.now()
        expired = []
        
        for ip, route in self.active_routes.items():
            if now > route.timestamp + timedelta(seconds=route.ttl):
                expired.append(route)
        
        # Remove rotas expiradas
        for route in expired:
            del self.active_routes[route.ip]
        
        return expired
    
    def remove_route(self, ip: str):
        """Remove rota manualmente"""
        if ip in self.active_routes:
            del self.active_routes[ip]


class BGPFlowController:
    """Controlador principal do fluxo BGP"""
    
    def __init__(self, kafka_config: Dict, exabgp_config: Dict):
        self.kafka_config = kafka_config
        self.exabgp_config = exabgp_config
        
        # Componentes
        self.flowspec_generator = BGPFlowspecGenerator()
        self.blackhole_generator = BGPBlackholeGenerator()
        self.exabgp_controller = ExaBGPController(exabgp_config.get('config_path'))
        self.mitigation_strategy = MitigationStrategy()
        self.ttl_manager = TTLManager()
        
        # Kafka
        self.consumer = KafkaConsumer(
            'malicious-ips',
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='bgp-controller'
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        logger.info("BGP Flow Controller inicializado")
    
    def process_malicious_ip(self, malicious_ip: Dict):
        """Processa IP malicioso e gera anúncio BGP"""
        try:
            ip = malicious_ip['ip']
            attack_type = malicious_ip.get('attack_type', 'unknown')
            threat_score = malicious_ip.get('threat_score', 0.5)
            ttl = malicious_ip.get('ttl', 3600)
            
            # Determinar estratégia de mitigação
            strategy = self.mitigation_strategy.get_strategy(attack_type, threat_score)
            
            # Gerar comando BGP baseado na estratégia
            if strategy == 'blackhole':
                bgp_command = self.blackhole_generator.generate_blackhole_route(malicious_ip)
                route_type = 'blackhole'
            else:
                bgp_command = self.flowspec_generator.generate_flowspec_rule(malicious_ip)
                route_type = 'flowspec'
            
            # Criar objeto de rota
            route = BGPRoute(
                ip=ip,
                prefix_length=32,
                next_hop=self.blackhole_generator.blackhole_next_hop,
                community=self.blackhole_generator.blackhole_community,
                action='announce',
                route_type=route_type,
                timestamp=datetime.now(),
                ttl=ttl,
                attack_type=attack_type,
                threat_score=threat_score
            )
            
            # Enviar comando BGP
            self.exabgp_controller.send_bgp_command(bgp_command)
            
            # Adicionar ao TTL manager
            self.ttl_manager.add_route(route)
            
            # Publicar no Kafka para logs/monitoramento
            bgp_update = {
                'action': 'announce',
                'ip': ip,
                'route_type': route_type,
                'strategy': strategy,
                'attack_type': attack_type,
                'threat_score': threat_score,
                'timestamp': datetime.now().isoformat(),
                'ttl': ttl
            }
            
            self.producer.send('bgp-updates', bgp_update)
            
            logger.info("Anúncio BGP processado",
                       ip=ip,
                       strategy=strategy,
                       route_type=route_type,
                       attack_type=attack_type)
            
        except Exception as e:
            logger.error("Erro ao processar IP malicioso", error=str(e))
    
    def withdraw_expired_routes(self):
        """Retira rotas expiradas"""
        expired_routes = self.ttl_manager.get_expired_routes()
        
        for route in expired_routes:
            try:
                # Gerar comando de retirada
                if route.route_type == 'blackhole':
                    withdraw_command = f"withdraw route {route.ip}/32"
                else:
                    withdraw_command = f"withdraw flow route {route.ip}/32"
                
                # Enviar comando de retirada
                self.exabgp_controller.send_bgp_command(withdraw_command)
                
                # Publicar no Kafka
                bgp_update = {
                    'action': 'withdraw',
                    'ip': route.ip,
                    'route_type': route.route_type,
                    'reason': 'ttl_expired',
                    'timestamp': datetime.now().isoformat()
                }
                
                self.producer.send('bgp-updates', bgp_update)
                
                logger.info("Rota BGP retirada por TTL",
                           ip=route.ip,
                           route_type=route.route_type)
                
            except Exception as e:
                logger.error("Erro ao retirar rota BGP", ip=route.ip, error=str(e))
    
    async def start_processing(self):
        """Inicia o processamento principal"""
        # Iniciar ExaBGP
        if not self.exabgp_controller.start_exabgp():
            logger.error("Falha ao iniciar ExaBGP")
            return
        
        logger.info("Iniciando processamento de IPs maliciosos")
        
        # Task para TTL management
        asyncio.create_task(self.ttl_management_loop())
        
        # Loop principal
        for message in self.consumer:
            try:
                malicious_ip = message.value
                self.process_malicious_ip(malicious_ip)
            except Exception as e:
                logger.error("Erro no processamento de mensagem", error=str(e))
    
    async def ttl_management_loop(self):
        """Loop de gerenciamento de TTL"""
        while True:
            try:
                self.withdraw_expired_routes()
                await asyncio.sleep(60)  # Verificar a cada minuto
            except Exception as e:
                logger.error("Erro no gerenciamento de TTL", error=str(e))
                await asyncio.sleep(60)


def create_exabgp_config():
    """Cria configuração básica do ExaBGP"""
    config = """
neighbor 192.168.1.1 {
    router-id 192.168.1.100;
    local-address 192.168.1.100;
    local-as 65001;
    peer-as 65000;
    
    family {
        ipv4 unicast;
        ipv4 flow;
    }
    
    process announce-routes {
        run /usr/bin/socat stdin UNIX-CONNECT:/var/run/exabgp/exabgp.in;
    }
}
"""
    
    os.makedirs("/etc/exabgp", exist_ok=True)
    with open("/etc/exabgp/exabgp.conf", "w") as f:
        f.write(config)
    
    logger.info("Configuração ExaBGP criada")


if __name__ == "__main__":
    # Configuração
    kafka_config = {
        'bootstrap_servers': ['kafka:29092']
    }
    
    exabgp_config = {
        'config_path': '/etc/exabgp/exabgp.conf'
    }
    
    # Criar configuração ExaBGP se não existir
    if not os.path.exists(exabgp_config['config_path']):
        create_exabgp_config()
    
    # Inicializar controlador
    controller = BGPFlowController(kafka_config, exabgp_config)
    
    # Executar processamento
    asyncio.run(controller.start_processing())
