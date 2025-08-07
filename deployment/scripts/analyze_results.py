#!/usr/bin/env python3
"""
Análise Comparativa dos Resultados de Otimização
Compara modelos antes e depois da otimização
"""

import json
import pandas as pd
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_results():
    """Carrega todos os resultados disponíveis"""
    results = {}
    models_dir = Path("./models")
    
    # Resultados básicos (hybrid)
    basic_dir = models_dir / "hybrid"
    if (basic_dir / "performance_metrics.json").exists():
        with open(basic_dir / "performance_metrics.json", 'r') as f:
            results['basic'] = json.load(f)
    
    # Resultados avançados (hybrid_advanced)  
    advanced_dir = models_dir / "hybrid_advanced"
    if (advanced_dir / "performance_advanced.json").exists():
        with open(advanced_dir / "performance_advanced.json", 'r') as f:
            results['advanced'] = json.load(f)
            
    # Resultados da otimização simplificada
    if (advanced_dir / "optimization_summary_simple.json").exists():
        with open(advanced_dir / "optimization_summary_simple.json", 'r') as f:
            results['optimized_simple'] = json.load(f)
    
    return results

def create_comparison_table(results):
    """Cria tabela comparativa dos resultados"""
    comparison_data = []
    
    for version, data in results.items():
        if version == 'basic':
            # Formato básico (hybrid)
            if 'ddos_specialist' in data:
                ddos_data = data['ddos_specialist']
                comparison_data.append({
                    'Version': 'Básico',
                    'Model': 'XGBoost (DDoS)',
                    'F1_Score': ddos_data.get('xgboost_f1', 0),
                    'Precision': 'N/A',
                    'Recall': 'N/A',
                    'Features': '50 (básicas)',
                    'Type': 'DDoS Specialist'
                })
                
                comparison_data.append({
                    'Version': 'Básico',
                    'Model': 'Random Forest (DDoS)',
                    'F1_Score': ddos_data.get('random_forest_f1', 0),
                    'Precision': 'N/A',
                    'Recall': 'N/A', 
                    'Features': '50 (básicas)',
                    'Type': 'DDoS Specialist'
                })
            
            if 'general_detector' in data:
                gen_data = data['general_detector']
                comparison_data.append({
                    'Version': 'Básico',
                    'Model': 'Isolation Forest',
                    'F1_Score': gen_data.get('isolation_forest_f1', 0),
                    'Precision': 'N/A',
                    'Recall': 'N/A',
                    'Features': '50 (básicas)',
                    'Type': 'General Detector'
                })
        
        elif version == 'advanced':
            # Formato avançado (hybrid_advanced)
            if 'ddos_xgb_f1' in data:
                comparison_data.append({
                    'Version': 'Avançado',
                    'Model': 'XGBoost (DDoS)',
                    'F1_Score': data.get('ddos_xgb_f1', 0),
                    'Precision': data.get('ddos_xgb_precision', 'N/A'),
                    'Recall': data.get('ddos_xgb_recall', 'N/A'),
                    'Features': '89 (feature engineering)',
                    'Type': 'DDoS Specialist'
                })
            
            if 'ddos_rf_f1' in data:
                comparison_data.append({
                    'Version': 'Avançado', 
                    'Model': 'Random Forest (DDoS)',
                    'F1_Score': data.get('ddos_rf_f1', 0),
                    'Precision': 'N/A',
                    'Recall': 'N/A',
                    'Features': '89 (feature engineering)',
                    'Type': 'DDoS Specialist'
                })
            
            if 'general_if_f1' in data:
                comparison_data.append({
                    'Version': 'Avançado',
                    'Model': 'Isolation Forest',
                    'F1_Score': data.get('general_if_f1', 0),
                    'Precision': 'N/A',
                    'Recall': 'N/A',
                    'Features': '89 (feature engineering)', 
                    'Type': 'General Detector'
                })
            
            if 'general_xgb_f1' in data:
                comparison_data.append({
                    'Version': 'Avançado',
                    'Model': 'XGBoost (General)',
                    'F1_Score': data.get('general_xgb_f1', 0),
                    'Precision': data.get('general_xgb_precision', 'N/A'),
                    'Recall': data.get('general_xgb_recall', 'N/A'),
                    'Features': '89 (feature engineering)',
                    'Type': 'General Detector'
                })
        
        elif version == 'optimized_simple':
            # Formato otimizado
            if 'results' in data:
                for model_name, model_data in data['results'].items():
                    if 'metrics' in model_data:
                        metrics = model_data['metrics']
                        model_type = 'General Detector' if 'Forest' in model_name else 'Unified'
                        comparison_data.append({
                            'Version': 'Otimizado',
                            'Model': model_name,
                            'F1_Score': metrics['f1_score'],
                            'Precision': metrics.get('precision', 'N/A'),
                            'Recall': metrics.get('recall', 'N/A'),
                            'Features': '89 (otimizado)',
                            'Type': model_type
                        })
    
    return pd.DataFrame(comparison_data)

def analyze_improvements(df):
    """Analisa melhorias entre versões"""
    logger.info("ANÁLISE DE MELHORIAS:")
    
    models = df['Model'].unique()
    
    for model in models:
        model_data = df[df['Model'] == model].copy()
        model_data = model_data.sort_values('Version')
        
        logger.info(f"\n🔍 {model}:")
        
        for _, row in model_data.iterrows():
            f1 = row['F1_Score']
            precision = row['Precision']
            recall = row['Recall']
            
            if isinstance(f1, (int, float)):
                f1_percent = f1 * 100 if f1 <= 1 else f1
                
                # Formatar precision e recall
                if isinstance(precision, (int, float)):
                    prec_str = f"{precision:.4f}"
                else:
                    prec_str = str(precision)
                    
                if isinstance(recall, (int, float)):
                    rec_str = f"{recall:.4f}"
                else:
                    rec_str = str(recall)
                
                logger.info(f"  {row['Version']:10} - F1: {f1_percent:6.2f}% | Precision: {prec_str} | Recall: {rec_str}")
            else:
                logger.info(f"  {row['Version']:10} - F1: N/A")
        
        # Calcular melhoria
        if len(model_data) > 1:
            f1_scores = model_data['F1_Score'].tolist()
            if all(isinstance(x, (int, float)) for x in f1_scores):
                improvement = ((f1_scores[-1] - f1_scores[0]) / f1_scores[0]) * 100
                logger.info(f"  📈 Melhoria: {improvement:+.2f}%")

def main():
    """Função principal"""
    try:
        logger.info("Iniciando análise comparativa dos resultados...")
        
        # Carregar resultados
        results = load_results()
        logger.info(f"Carregados {len(results)} conjuntos de resultados")
        
        if not results:
            logger.warning("Nenhum resultado encontrado!")
            return False
        
        # Criar tabela comparativa
        df = create_comparison_table(results)
        
        if df.empty:
            logger.warning("Nenhum dado válido para comparação!")
            return False
        
        # Mostrar tabela
        logger.info("\nTABELA COMPARATIVA:")
        print("\n" + "="*120)
        print(df.to_string(index=False))
        print("="*120)
        
        # Análise de melhorias
        analyze_improvements(df)
        
        # Salvar tabela
        output_file = Path("./models/hybrid_advanced/comparison_analysis.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"\nAnálise salva em: {output_file}")
        
        # Resumo final
        logger.info("\nRESUMO FINAL:")
        xgb_data = df[df['Model'] == 'XGBoost']
        if_data = df[df['Model'] == 'Isolation Forest']
        
        if not xgb_data.empty:
            best_xgb = xgb_data.loc[xgb_data['F1_Score'].idxmax()]
            logger.info(f"  🏆 Melhor XGBoost: {best_xgb['Version']} - F1: {best_xgb['F1_Score']*100:.2f}%")
        
        if not if_data.empty:
            best_if = if_data.loc[if_data['F1_Score'].idxmax()]
            logger.info(f"  🏆 Melhor Isolation Forest: {best_if['Version']} - F1: {best_if['F1_Score']*100:.2f}%")
        
        logger.info("Análise comparativa concluída!")
        return True
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
