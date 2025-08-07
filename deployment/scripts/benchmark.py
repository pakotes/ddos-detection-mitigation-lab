"""
Benchmark e Avaliação de Performance do Sistema DDoS Mitigation

Este módulo implementa:
1. Medição de latência ponta-a-ponta
2. Testes de throughput
3. Análise de tempo de resposta BGP
4. Geração de relatórios de performance
"""

import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import statistics
from dataclasses import dataclass, asdict
from kafka import KafkaProducer, KafkaConsumer
import structlog

logger = structlog.get_logger()


@dataclass
class PerformanceMetric:
    """Métrica de performance"""
    timestamp: datetime
    metric_type: str
    value: float
    unit: str
    metadata: Dict


class LatencyTracker:
    """Rastreador de latência ponta-a-ponta"""
    
    def __init__(self):
        self.pending_flows = {}  # flow_id -> start_time
        self.completed_latencies = []
        self.lock = threading.Lock()
    
    def start_flow_tracking(self, flow_id: str) -> float:
        """Inicia rastreamento de um fluxo"""
        start_time = time.time()
        with self.lock:
            self.pending_flows[flow_id] = start_time
        return start_time
    
    def complete_flow_tracking(self, flow_id: str) -> Optional[float]:
        """Completa rastreamento e retorna latência"""
        end_time = time.time()
        with self.lock:
            start_time = self.pending_flows.pop(flow_id, None)
            if start_time:
                latency = end_time - start_time
                self.completed_latencies.append(latency)
                return latency
        return None
    
    def get_latency_stats(self) -> Dict:
        """Retorna estatísticas de latência"""
        with self.lock:
            if not self.completed_latencies:
                return {}
            
            return {
                'count': len(self.completed_latencies),
                'mean': statistics.mean(self.completed_latencies),
                'median': statistics.median(self.completed_latencies),
                'min': min(self.completed_latencies),
                'max': max(self.completed_latencies),
                'std_dev': statistics.stdev(self.completed_latencies) if len(self.completed_latencies) > 1 else 0,
                'p95': self._percentile(self.completed_latencies, 95),
                'p99': self._percentile(self.completed_latencies, 99)
            }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calcula percentil"""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile / 100
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c


class ThroughputMeter:
    """Medidor de throughput"""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.events = []
        self.lock = threading.Lock()
    
    def record_event(self, event_type: str):
        """Registra um evento"""
        timestamp = time.time()
        with self.lock:
            self.events.append({'timestamp': timestamp, 'type': event_type})
            # Limpar eventos antigos
            cutoff_time = timestamp - self.window_size
            self.events = [e for e in self.events if e['timestamp'] > cutoff_time]
    
    def get_throughput_stats(self) -> Dict:
        """Retorna estatísticas de throughput"""
        with self.lock:
            current_time = time.time()
            
            # Contar eventos por tipo
            event_counts = {}
            for event in self.events:
                event_type = event['type']
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Calcular taxa por segundo
            throughput_stats = {}
            for event_type, count in event_counts.items():
                throughput_stats[f'{event_type}_per_second'] = count / self.window_size
            
            throughput_stats['total_events'] = len(self.events)
            throughput_stats['window_size_seconds'] = self.window_size
            
            return throughput_stats


class SystemBenchmark:
    """Benchmark completo do sistema"""
    
    def __init__(self, kafka_config: Dict):
        self.kafka_config = kafka_config
        self.latency_tracker = LatencyTracker()
        self.throughput_meter = ThroughputMeter()
        self.metrics = []
        
        # Kafka setup
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        self.consumer = KafkaConsumer(
            'bgp-updates',
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='benchmark-consumer',
            auto_offset_reset='latest'
        )
    
    async def run_latency_benchmark(self, num_flows: int = 1000) -> Dict:
        """Executa benchmark de latência"""
        logger.info("Iniciando benchmark de latência", num_flows=num_flows)
        
        # Iniciar consumidor BGP em thread separada
        bgp_consumer_thread = threading.Thread(
            target=self._consume_bgp_updates, 
            daemon=True
        )
        bgp_consumer_thread.start()
        
        # Gerar fluxos de teste
        start_time = time.time()
        
        for i in range(num_flows):
            flow_id = f"benchmark_flow_{i}"
            
            # Criar fluxo malicioso sintético
            malicious_flow = {
                'flow_id': flow_id,
                'timestamp': datetime.now().isoformat(),
                'source': 'benchmark',
                'event_type': 'flow',
                'src_ip': f'10.0.{i//256}.{i%256}',
                'dst_ip': '192.168.1.1',
                'src_port': 12345 + i,
                'dst_port': 80,
                'protocol': 6,
                'bytes': 1000000 + (i * 1000),  # Tráfego volumétrico
                'packets': 1000 + (i * 10),
                'duration': 1.0,
                'tcp_flags': 0x02  # SYN flood
            }
            
            # Iniciar rastreamento
            self.latency_tracker.start_flow_tracking(flow_id)
            
            # Enviar para processamento
            self.producer.send('traffic-logs', malicious_flow)
            self.throughput_meter.record_event('flow_generated')
            
            # Pequena pausa para não sobrecarregar
            if i % 100 == 0:
                await asyncio.sleep(0.1)
                logger.info(f"Progresso benchmark: {i}/{num_flows}")
        
        # Aguardar processamento
        await asyncio.sleep(30)
        
        total_time = time.time() - start_time
        
        # Coletar estatísticas
        latency_stats = self.latency_tracker.get_latency_stats()
        throughput_stats = self.throughput_meter.get_throughput_stats()
        
        benchmark_results = {
            'test_type': 'latency_benchmark',
            'timestamp': datetime.now().isoformat(),
            'test_duration_seconds': total_time,
            'flows_generated': num_flows,
            'latency_stats': latency_stats,
            'throughput_stats': throughput_stats
        }
        
        logger.info("Benchmark de latência concluído", results=benchmark_results)
        
        return benchmark_results
    
    def _consume_bgp_updates(self):
        """Consome updates BGP para medir latência"""
        for message in self.consumer:
            try:
                bgp_update = message.value
                
                # Extrair flow_id se presente
                if 'metadata' in bgp_update and 'flow_id' in bgp_update['metadata']:
                    flow_id = bgp_update['metadata']['flow_id']
                    latency = self.latency_tracker.complete_flow_tracking(flow_id)
                    
                    if latency:
                        self.throughput_meter.record_event('bgp_announced')
                        logger.debug("BGP update processado", flow_id=flow_id, latency=latency)
                
            except Exception as e:
                logger.error("Erro ao processar BGP update", error=str(e))
    
    async def run_stress_test(self, duration_seconds: int = 300, flows_per_second: int = 100) -> Dict:
        """Executa teste de stress"""
        logger.info("Iniciando teste de stress", 
                   duration=duration_seconds, 
                   flows_per_second=flows_per_second)
        
        start_time = time.time()
        flow_count = 0
        
        while time.time() - start_time < duration_seconds:
            batch_start = time.time()
            
            # Gerar lote de fluxos
            for i in range(flows_per_second):
                flow_id = f"stress_flow_{flow_count}_{i}"
                
                stress_flow = {
                    'flow_id': flow_id,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'stress_test',
                    'event_type': 'flow',
                    'src_ip': f'172.16.{flow_count%256}.{i%256}',
                    'dst_ip': '192.168.1.1',
                    'src_port': 1024 + (i % 64511),
                    'dst_port': 80,
                    'protocol': 6,
                    'bytes': 500000 + (i * 100),
                    'packets': 500 + (i * 5),
                    'duration': 0.5,
                    'tcp_flags': 0x02
                }
                
                self.producer.send('traffic-logs', stress_flow)
                self.throughput_meter.record_event('stress_flow')
            
            flow_count += 1
            
            # Aguardar para manter taxa desejada
            batch_time = time.time() - batch_start
            if batch_time < 1.0:
                await asyncio.sleep(1.0 - batch_time)
            
            # Log de progresso
            if flow_count % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Stress test progresso: {elapsed:.1f}s/{duration_seconds}s")
        
        total_time = time.time() - start_time
        total_flows = flow_count * flows_per_second
        
        stress_results = {
            'test_type': 'stress_test',
            'timestamp': datetime.now().isoformat(),
            'test_duration_seconds': total_time,
            'target_flows_per_second': flows_per_second,
            'total_flows_generated': total_flows,
            'actual_flows_per_second': total_flows / total_time,
            'throughput_stats': self.throughput_meter.get_throughput_stats()
        }
        
        logger.info("Teste de stress concluído", results=stress_results)
        
        return stress_results
    
    def run_memory_usage_test(self) -> Dict:
        """Testa uso de memória"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        memory_stats = {
            'rss_mb': process.memory_info().rss / 1024 / 1024,
            'vms_mb': process.memory_info().vms / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'num_threads': process.num_threads(),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
        }
        
        return memory_stats
    
    def generate_performance_report(self, test_results: List[Dict]) -> Dict:
        """Gera relatório de performance"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'python_version': f"{__import__('sys').version}",
                'kafka_bootstrap_servers': self.kafka_config['bootstrap_servers']
            },
            'test_results': test_results,
            'summary': {
                'total_tests': len(test_results),
                'test_types': list(set(r.get('test_type', 'unknown') for r in test_results))
            }
        }
        
        # Resumo de latências se disponível
        latency_tests = [r for r in test_results if 'latency_stats' in r]
        if latency_tests:
            all_latencies = []
            for test in latency_tests:
                if test['latency_stats']:
                    all_latencies.extend([
                        test['latency_stats'].get('mean', 0),
                        test['latency_stats'].get('p95', 0),
                        test['latency_stats'].get('p99', 0)
                    ])
            
            if all_latencies:
                report['summary']['overall_latency'] = {
                    'mean_of_means': statistics.mean(all_latencies),
                    'best_latency': min(all_latencies),
                    'worst_latency': max(all_latencies)
                }
        
        return report


async def main():
    """Função principal para executar benchmarks"""
    kafka_config = {
        'bootstrap_servers': ['localhost:9092']
    }
    
    benchmark = SystemBenchmark(kafka_config)
    test_results = []
    
    try:
        # Benchmark de latência
        logger.info("Executando benchmark de latência...")
        latency_results = await benchmark.run_latency_benchmark(500)
        test_results.append(latency_results)
        
        # Aguardar sistema se estabilizar
        await asyncio.sleep(30)
        
        # Teste de stress
        logger.info("Executando teste de stress...")
        stress_results = await benchmark.run_stress_test(120, 50)  # 2 minutos, 50 flows/s
        test_results.append(stress_results)
        
        # Teste de memória
        memory_results = benchmark.run_memory_usage_test()
        test_results.append({
            'test_type': 'memory_usage',
            'timestamp': datetime.now().isoformat(),
            'memory_stats': memory_results
        })
        
        # Gerar relatório final
        report = benchmark.generate_performance_report(test_results)
        
        # Salvar relatório
        report_file = f"/app/benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Benchmark concluído", report_file=report_file)
        
        return report
        
    except Exception as e:
        logger.error("Erro durante benchmark", error=str(e))
        raise
    finally:
        benchmark.producer.close()


if __name__ == "__main__":
    asyncio.run(main())
