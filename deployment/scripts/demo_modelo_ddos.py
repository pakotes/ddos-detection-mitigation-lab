#!/usr/bin/env python3
"""
Exemplo de uso do modelo DDoS otimizado
Demonstra como integrar o modelo em aplicações
"""

import pickle
import numpy as np
import pandas as pd
import time
from pathlib import Path

def load_ddos_model():
    """Carregar modelo DDoS otimizado"""
    # Tentar diferentes localizações de modelo (relativo à pasta scripts)
    base_path = Path(__file__).parent.parent.parent  # Voltar para raiz do projeto
    possible_paths = [
        base_path / 'src' / 'models' / 'optimized' / 'ddos_optimized_model.pkl',
        base_path / 'src' / 'models' / 'hybrid_advanced' / 'hybrid_advanced_models.pkl',
        base_path / 'src' / 'models' / 'hybrid' / 'hybrid_models.pkl'
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break
    
    if not model_path:
        print("❌ Nenhum modelo encontrado!")
        print("Execute primeiro: ./deployment/scripts/make.sh train")
        print("Locais procurados:")
        for path in possible_paths:
            print(f"  - {path}")
        return None
    
    print(f"📦 Carregando modelo de: {model_path.name}")
    
    with open(model_path, 'rb') as f:
        package = pickle.load(f)
    
    # Verificar estrutura do modelo
    if isinstance(package, dict):
        model_info = package
    else:
        # Modelo simples sem metadata
        model_info = {'model': package, 'model_type': 'Unknown', 'n_features': 'Unknown'}
    
    print(f"✅ Modelo carregado: {model_info.get('model_type', 'Unknown')}")
    
    if 'n_features' in model_info:
        print(f"   Features: {model_info['n_features']}")
    if 'performance' in model_info:
        perf = model_info['performance']
        if isinstance(perf, dict) and 'accuracy' in perf:
            print(f"   Accuracy: {perf['accuracy']:.3f}")
    
    return model_info

def simulate_network_traffic(n_samples=1000):
    """Simular tráfego de rede para demonstração"""
    print(f"\n📡 Simulando {n_samples} pacotes de rede...")
    
    np.random.seed(42)
    
    # Simular 50 features de rede (igual aos dados de treino)
    network_data = np.random.randn(n_samples, 50)
    
    # Criar alguns padrões "suspeitos" para demonstração
    # 10% dos dados serão "ataques simulados"
    n_attacks = int(n_samples * 0.1)
    attack_indices = np.random.choice(n_samples, n_attacks, replace=False)
    
    # Tornar ataques detectáveis (amplificar certas features)
    network_data[attack_indices, :10] *= 3  # Features que modelo considera importantes
    
    # Labels reais (para comparação)
    true_labels = np.zeros(n_samples)
    true_labels[attack_indices] = 1
    
    return network_data, true_labels

def detect_ddos_attacks(package, network_data):
    """Detectar ataques DDoS usando modelo otimizado"""
    print("\n🔍 Analisando tráfego...")
    
    # Extrair componentes do modelo (com fallbacks)
    if isinstance(package, dict):
        model = package.get('model', package)
        feature_selector = package.get('feature_selector')
        scaler = package.get('scaler')
    else:
        model = package
        feature_selector = None
        scaler = None
    
    # Preprocessar dados
    start_time = time.time()
    
    processed_data = network_data.copy()
    
    if scaler:
        processed_data = scaler.transform(processed_data)
    
    # Selecionar features importantes (se disponível)
    if feature_selector:
        processed_data = feature_selector.transform(processed_data)
    
    # Predição
    predictions = model.predict(processed_data)
    
    # Probabilidades (se disponível)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(processed_data)[:, 1]
    else:
        probabilities = predictions.astype(float)
    
    processing_time = time.time() - start_time
    
    return predictions, probabilities, processing_time

def analyze_results(predictions, probabilities, true_labels, processing_time):
    """Analisar resultados da detecção"""
    print(f"\n📊 RESULTADOS DA ANÁLISE:")
    print(f"   Tempo de processamento: {processing_time:.4f}s")
    print(f"   Velocidade: {len(predictions)/processing_time:,.0f} amostras/segundo")
    
    # Estatísticas de detecção
    n_detected = predictions.sum()
    n_total = len(predictions)
    detection_rate = n_detected / n_total
    
    print(f"\n🎯 DETECÇÕES:")
    print(f"   Total analisado: {n_total:,} amostras")
    print(f"   Ataques detectados: {n_detected} ({detection_rate:.1%})")
    
    # Ataques de alta confiança
    high_confidence = (probabilities > 0.8).sum()
    print(f"   Alta confiança (>80%): {high_confidence}")
    
    # Comparar com labels reais (se disponível)
    if true_labels is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        
        print(f"\n✅ MÉTRICAS DE QUALIDADE:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        
        # Matriz de confusão simplificada
        tp = ((predictions == 1) & (true_labels == 1)).sum()
        fp = ((predictions == 1) & (true_labels == 0)).sum()
        tn = ((predictions == 0) & (true_labels == 0)).sum()
        fn = ((predictions == 0) & (true_labels == 1)).sum()
        
        print(f"\n📈 DETALHES:")
        print(f"   Verdadeiros Positivos: {tp}")
        print(f"   Falsos Positivos: {fp}")
        print(f"   Verdadeiros Negativos: {tn}")
        print(f"   Falsos Negativos: {fn}")
    
    return {
        'detection_rate': detection_rate,
        'processing_time': processing_time,
        'samples_per_second': len(predictions)/processing_time,
        'high_confidence_detections': high_confidence
    }

def real_time_simulation(package, duration=30):
    """Simular detecção em tempo real"""
    print(f"\n⚡ SIMULAÇÃO TEMPO REAL ({duration}s)")
    print("Processando lotes de tráfego continuamente...")
    
    start_time = time.time()
    total_processed = 0
    total_attacks = 0
    batch_times = []
    
    while time.time() - start_time < duration:
        # Lote de 100 amostras
        batch_data, _ = simulate_network_traffic(100)
        
        batch_start = time.time()
        predictions, _, _ = detect_ddos_attacks(package, batch_data)
        batch_time = time.time() - batch_start
        
        batch_times.append(batch_time)
        total_processed += len(predictions)
        total_attacks += predictions.sum()
        
        # Pausa pequena para simular chegada de dados
        time.sleep(0.05)
    
    elapsed = time.time() - start_time
    
    print(f"\n🏁 RESULTADO DA SIMULAÇÃO:")
    print(f"   Duração: {elapsed:.1f}s")
    print(f"   Amostras processadas: {total_processed:,}")
    print(f"   Throughput: {total_processed/elapsed:,.0f} amostras/s")
    print(f"   Ataques detectados: {total_attacks}")
    print(f"   Tempo médio por lote: {np.mean(batch_times):.4f}s")
    print(f"   Latência máxima: {np.max(batch_times):.4f}s")

def main():
    """Função principal"""
    print("🛡️ DEMONSTRAÇÃO - Modelo DDoS Otimizado")
    print("="*50)
    
    # 1. Carregar modelo
    package = load_ddos_model()
    if not package:
        return 1
    
    # 2. Simular dados de rede
    network_data, true_labels = simulate_network_traffic(5000)
    
    # 3. Detectar ataques
    predictions, probabilities, processing_time = detect_ddos_attacks(package, network_data)
    
    # 4. Analisar resultados
    results = analyze_results(predictions, probabilities, true_labels, processing_time)
    
    # 5. Simulação tempo real
    real_time_simulation(package, duration=10)
    
    print(f"\n🎉 DEMONSTRAÇÃO CONCLUÍDA!")
    print(f"💡 O modelo está pronto para integração em sistemas de produção.")
    
    return 0

if __name__ == "__main__":
    exit(main())
