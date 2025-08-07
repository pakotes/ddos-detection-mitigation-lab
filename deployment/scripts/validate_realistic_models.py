#!/usr/bin/env python3
"""
Validação melhorada para datasets realistas
Verifica se os resultados são realistas (não overfitting)
"""

import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def validate_realistic_results():
    """Validar resultados dos modelos realistas"""
    logger.info("Validando resultados dos modelos REALISTAS...")
    
    models_dir = Path(__file__).parent.parent.parent / 'src' / 'models' / 'realistic'
    
    if not models_dir.exists():
        logger.error("Diretório de modelos realistas não encontrado!")
        logger.info("Execute primeiro: ddos-train-realistic")
        return False
    
    # Verificar arquivos de resultados
    result_files = list(models_dir.glob('results_*.json'))
    
    if not result_files:
        logger.error("Nenhum arquivo de resultados encontrado!")
        return False
    
    logger.info(f"\n{'='*60}")
    logger.info("VALIDAÇÃO DE QUALIDADE DOS MODELOS REALISTAS")
    logger.info(f"{'='*60}")
    
    all_results = {}
    
    for result_file in result_files:
        dataset_name = result_file.stem.replace('results_', '')
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        all_results[dataset_name] = results
        
        logger.info(f"\n--- DATASET: {dataset_name.upper()} ---")
        
        # Analisar cada modelo
        models = results.get('models', {})
        
        for model_name, metrics in models.items():
            accuracy = metrics.get('accuracy', 0)
            f1 = metrics.get('f1_score', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            
            logger.info(f"{model_name:20}: Acc={accuracy:.3f}, F1={f1:.3f}, Prec={precision:.3f}, Rec={recall:.3f}")
            
            # Avaliar qualidade
            if accuracy >= 1.0:
                logger.warning(f"  ⚠️  OVERFITTING SUSPEITO: {model_name} com {accuracy:.3f} accuracy")
            elif accuracy >= 0.99:
                logger.info(f"  ✅ BOM: {model_name} com {accuracy:.3f} accuracy (realista)")
            elif accuracy >= 0.90:
                logger.info(f"  ✅ ACEITÁVEL: {model_name} com {accuracy:.3f} accuracy")
            else:
                logger.warning(f"  ⚠️  BAIXO: {model_name} com {accuracy:.3f} accuracy")
            
            # Verificar cross-validation se disponível
            cv_mean = metrics.get('cv_f1_mean')
            cv_std = metrics.get('cv_f1_std')
            
            if cv_mean is not None and cv_std is not None:
                logger.info(f"    CV F1: {cv_mean:.3f} ± {cv_std:.3f}")
                
                # Verificar consistência
                if abs(f1 - cv_mean) > 0.05:
                    logger.warning(f"    ⚠️  Grande diferença entre teste e CV")
            
            # Verificar confiança dos modelos
            conf_stats = metrics.get('confidence_stats', {})
            if conf_stats:
                mean_conf = conf_stats.get('mean_confidence', 0)
                high_conf_ratio = conf_stats.get('high_confidence_ratio', 0)
                
                logger.info(f"    Confiança média: {mean_conf:.3f}")
                logger.info(f"    % alta confiança: {high_conf_ratio:.1%}")
                
                if high_conf_ratio > 0.95:
                    logger.warning(f"    ⚠️  Muita confiança alta ({high_conf_ratio:.1%}) - possível overfitting")
    
    # Resumo geral
    logger.info(f"\n{'='*60}")
    logger.info("RESUMO GERAL")
    logger.info(f"{'='*60}")
    
    total_models = 0
    good_models = 0
    overfitting_models = 0
    
    for dataset_name, results in all_results.items():
        best_acc = results.get('summary', {}).get('best_accuracy', 0)
        best_f1 = results.get('summary', {}).get('best_f1', 0)
        
        logger.info(f"{dataset_name:15}: Melhor Acc={best_acc:.3f}, Melhor F1={best_f1:.3f}")
        
        models = results.get('models', {})
        for model_name, metrics in models.items():
            total_models += 1
            accuracy = metrics.get('accuracy', 0)
            
            if accuracy >= 1.0:
                overfitting_models += 1
            elif accuracy >= 0.90:
                good_models += 1
    
    logger.info(f"\nEstatísticas:")
    logger.info(f"  Total de modelos: {total_models}")
    logger.info(f"  Modelos bons (>90%): {good_models}")
    logger.info(f"  Modelos com overfitting (100%): {overfitting_models}")
    logger.info(f"  Taxa de sucesso: {(good_models/total_models):.1%}")
    
    # Avaliação final
    if overfitting_models == 0:
        logger.info(f"\n🎉 EXCELENTE: Nenhum modelo com overfitting!")
        logger.info(f"   Os datasets realistas funcionaram perfeitamente.")
    elif overfitting_models <= total_models * 0.2:
        logger.info(f"\n✅ BOM: Apenas {overfitting_models} de {total_models} modelos com overfitting.")
    else:
        logger.warning(f"\n⚠️  ATENÇÃO: {overfitting_models} de {total_models} modelos com overfitting.")
        logger.warning(f"   Considere aumentar o overlap nos datasets.")
    
    return True

def analyze_dataset_quality():
    """Analisar qualidade dos datasets criados"""
    logger.info(f"\n{'='*60}")
    logger.info("ANÁLISE DE QUALIDADE DOS DATASETS")
    logger.info(f"{'='*60}")
    
    datasets_dir = Path(__file__).parent.parent.parent / 'src' / 'datasets' / 'realistic'
    
    if not datasets_dir.exists():
        logger.warning("Diretório de datasets realistas não encontrado!")
        return False
    
    # Analisar cada dataset
    for dataset_name in ['cicddos', 'unsw', 'integrated']:
        metadata_file = datasets_dir / f'metadata_{dataset_name}.json'
        
        if not metadata_file.exists():
            logger.warning(f"Metadados não encontrados para {dataset_name}")
            continue
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"\n--- DATASET: {dataset_name.upper()} ---")
        logger.info(f"Amostras totais: {metadata.get('total_samples', 'N/A'):,}")
        logger.info(f"Features: {metadata.get('n_features', 'N/A')}")
        logger.info(f"Classes: {metadata.get('n_classes', 'N/A')}")
        logger.info(f"Taxa de ataques: {metadata.get('attack_ratio', 0):.1%}")
        logger.info(f"Taxa normal: {metadata.get('normal_ratio', 0):.1%}")
        
        # Verificar balanceamento
        attack_ratio = metadata.get('attack_ratio', 0)
        if attack_ratio >= 1.0:
            logger.error(f"  ❌ PROBLEMA: 100% ataques, sem amostras normais!")
        elif attack_ratio >= 0.95:
            logger.warning(f"  ⚠️  Muito desbalanceado: {attack_ratio:.1%} ataques")
        elif 0.3 <= attack_ratio <= 0.8:
            logger.info(f"  ✅ Bem balanceado: {attack_ratio:.1%} ataques")
        else:
            logger.warning(f"  ⚠️  Balanceamento questionável: {attack_ratio:.1%} ataques")
        
        # Verificar melhorias implementadas
        improvements = metadata.get('improvements', [])
        if improvements:
            logger.info(f"  Melhorias implementadas: {len(improvements)}")
            for improvement in improvements:
                logger.info(f"    - {improvement}")
    
    return True

def main():
    """Função principal"""
    logger.info("Iniciando validação completa dos modelos realistas...")
    
    try:
        # 1. Validar resultados dos modelos
        if not validate_realistic_results():
            return False
        
        # 2. Analisar qualidade dos datasets
        if not analyze_dataset_quality():
            logger.warning("Análise de datasets falhou, mas continuando...")
        
        logger.info(f"\n{'='*60}")
        logger.info("VALIDAÇÃO CONCLUÍDA")
        logger.info(f"{'='*60}")
        logger.info("✅ Validação dos modelos realistas concluída com sucesso!")
        logger.info("📊 Verifique os logs acima para detalhes da qualidade.")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro na validação: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
