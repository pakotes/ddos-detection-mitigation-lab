"""
Agregador de Janelas de Tempo para ML Processor

Este módulo implementa a agregação de dados em janelas de 5 segundos
conforme especificado na arquitetura do sistema.
"""

import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np


class TimeWindowAggregator:
    """Agregador de dados em janelas de tempo de 5 segundos"""
    
    def __init__(self, window_size_seconds: int = 5):
        self.window_size = window_size_seconds
        self.windows = defaultdict(lambda: {
            'flows': [],
            'start_time': None,
            'end_time': None
        })
        self.completed_windows = deque(maxlen=100)  # Manter últimas 100 janelas
        self.lock = threading.Lock()
        
        # Iniciar thread de limpeza
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_windows, daemon=True)
        self.cleanup_thread.start()
    
    def add_flow(self, flow_data: Dict) -> Optional[Dict]:
        """Adiciona fluxo à janela apropriada e retorna janela completa se pronta"""
        timestamp = datetime.fromisoformat(flow_data['timestamp'].replace('Z', '+00:00'))
        window_key = self._get_window_key(timestamp)
        
        with self.lock:
            window = self.windows[window_key]
            
            if window['start_time'] is None:
                window['start_time'] = timestamp
                window['end_time'] = timestamp + timedelta(seconds=self.window_size)
            
            window['flows'].append(flow_data)
            
            # Verificar se janela está completa
            if timestamp >= window['end_time']:
                completed_window = self._finalize_window(window_key)
                return completed_window
        
        return None
    
    def _get_window_key(self, timestamp: datetime) -> int:
        """Gera chave única para a janela baseada no timestamp"""
        return int(timestamp.timestamp() // self.window_size)
    
    def _finalize_window(self, window_key: int) -> Dict:
        """Finaliza janela e gera features agregadas"""
        window = self.windows.pop(window_key)
        flows = window['flows']
        
        if not flows:
            return None
        
        # Agregar features da janela
        aggregated_features = self._aggregate_window_features(flows)
        
        window_data = {
            'window_key': window_key,
            'start_time': window['start_time'].isoformat(),
            'end_time': window['end_time'].isoformat(),
            'flow_count': len(flows),
            'features': aggregated_features,
            'raw_flows': flows
        }
        
        self.completed_windows.append(window_data)
        return window_data
    
    def _aggregate_window_features(self, flows: List[Dict]) -> Dict:
        """Agrega features volumétricas, comportamentais e de protocolo"""
        features = {
            # Features volumétricas
            'total_bytes': 0,
            'total_packets': 0,
            'total_flows': len(flows),
            'avg_bytes_per_flow': 0,
            'avg_packets_per_flow': 0,
            'bytes_per_second': 0,
            'packets_per_second': 0,
            'flows_per_second': 0,
            
            # Features comportamentais
            'unique_src_ips': set(),
            'unique_dst_ips': set(),
            'unique_src_ports': set(),
            'unique_dst_ports': set(),
            'syn_flag_count': 0,
            'ack_flag_count': 0,
            'fin_flag_count': 0,
            'rst_flag_count': 0,
            
            # Features de protocolo
            'tcp_flows': 0,
            'udp_flows': 0,
            'icmp_flows': 0,
            'port_80_flows': 0,
            'port_443_flows': 0,
            'port_53_flows': 0,
            'high_port_flows': 0,
            
            # Features estatísticas
            'bytes_std': 0,
            'packets_std': 0,
            'duration_mean': 0,
            'duration_std': 0,
        }
        
        bytes_list = []
        packets_list = []
        durations = []
        
        for flow in flows:
            # Volumétricas
            flow_bytes = flow.get('bytes', 0)
            flow_packets = flow.get('packets', 0)
            features['total_bytes'] += flow_bytes
            features['total_packets'] += flow_packets
            bytes_list.append(flow_bytes)
            packets_list.append(flow_packets)
            
            # Comportamentais
            features['unique_src_ips'].add(flow.get('src_ip', ''))
            features['unique_dst_ips'].add(flow.get('dst_ip', ''))
            features['unique_src_ports'].add(flow.get('src_port', 0))
            features['unique_dst_ports'].add(flow.get('dst_port', 0))
            
            # Flags TCP
            tcp_flags = flow.get('tcp_flags', 0)
            if tcp_flags & 0x02:  # SYN
                features['syn_flag_count'] += 1
            if tcp_flags & 0x10:  # ACK
                features['ack_flag_count'] += 1
            if tcp_flags & 0x01:  # FIN
                features['fin_flag_count'] += 1
            if tcp_flags & 0x04:  # RST
                features['rst_flag_count'] += 1
            
            # Protocolo
            protocol = flow.get('protocol', 0)
            if protocol == 6:  # TCP
                features['tcp_flows'] += 1
            elif protocol == 17:  # UDP
                features['udp_flows'] += 1
            elif protocol == 1:  # ICMP
                features['icmp_flows'] += 1
            
            # Portas específicas
            dst_port = flow.get('dst_port', 0)
            if dst_port == 80:
                features['port_80_flows'] += 1
            elif dst_port == 443:
                features['port_443_flows'] += 1
            elif dst_port == 53:
                features['port_53_flows'] += 1
            elif dst_port > 1024:
                features['high_port_flows'] += 1
            
            # Duração
            duration = flow.get('duration', 0)
            if duration > 0:
                durations.append(duration)
        
        # Calcular médias e estatísticas
        if features['total_flows'] > 0:
            features['avg_bytes_per_flow'] = features['total_bytes'] / features['total_flows']
            features['avg_packets_per_flow'] = features['total_packets'] / features['total_flows']
            features['bytes_per_second'] = features['total_bytes'] / self.window_size
            features['packets_per_second'] = features['total_packets'] / self.window_size
            features['flows_per_second'] = features['total_flows'] / self.window_size
            
            # Converter sets para contagens
            features['unique_src_ips'] = len(features['unique_src_ips'])
            features['unique_dst_ips'] = len(features['unique_dst_ips'])
            features['unique_src_ports'] = len(features['unique_src_ports'])
            features['unique_dst_ports'] = len(features['unique_dst_ports'])
            
            # Estatísticas de distribuição
            if bytes_list:
                features['bytes_std'] = np.std(bytes_list)
            if packets_list:
                features['packets_std'] = np.std(packets_list)
            if durations:
                features['duration_mean'] = np.mean(durations)
                features['duration_std'] = np.std(durations)
        
        return features
    
    def _cleanup_expired_windows(self):
        """Thread para limpeza de janelas expiradas"""
        while True:
            try:
                current_time = datetime.now()
                expired_keys = []
                
                with self.lock:
                    for key, window in self.windows.items():
                        if window['start_time'] and (current_time - window['start_time']).total_seconds() > self.window_size * 2:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self.windows.pop(key, None)
                
                time.sleep(self.window_size)
                
            except Exception as e:
                print(f"Erro na limpeza de janelas: {e}")
                time.sleep(self.window_size)
    
    def get_recent_windows(self, count: int = 10) -> List[Dict]:
        """Retorna as janelas mais recentes"""
        with self.lock:
            return list(self.completed_windows)[-count:]
    
    def get_window_statistics(self) -> Dict:
        """Retorna estatísticas das janelas"""
        with self.lock:
            return {
                'active_windows': len(self.windows),
                'completed_windows': len(self.completed_windows),
                'window_size_seconds': self.window_size
            }
