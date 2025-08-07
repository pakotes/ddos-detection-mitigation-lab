#!/usr/bin/env python3
"""
Script para verificar e validar os resultados dos modelos treinados
Analisa se 100% acurácia é realista ou indica overfitting
"""

import json
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def validate_models():
    """Validar modelos de forma simples e clara"""
    
    models_dir = Path(__file__).parent.parent.parent / 'src' / 'models' / 'clean'
    datasets_dir = Path(__file__).parent.parent.parent / 'src' / 'datasets' / 'clean'
    
    if not models_dir.exists():
        logger.error("❌ Modelos não encontrados. Execute 'ddos-train-clean' primeiro.")
        return False
    
    print("🔍 VALIDAÇÃO DOS MODELOS - Verificando 100% Acurácia")
    print("=" * 60)
    
    datasets = ['cicddos', 'unsw', 'integrated']
    all_perfect = True
    
    for dataset_name in datasets:
        print(f"\n📊 Dataset: {dataset_name.upper()}")
        print("-" * 30)
        
        try:
            # Carregar dados de teste
            X_test = np.load(datasets_dir / f'X_test_{dataset_name}.npy')
            y_test = np.load(datasets_dir / f'y_test_binary_{dataset_name}.npy')
            
            print(f"Amostras de teste: {len(X_test):,}")
            print(f"Ataques: {y_test.sum():,} ({y_test.mean():.1%})")
            print(f"Benignos: {(1-y_test).sum():,} ({(1-y_test.mean()):.1%})")
            
            # Verificar Random Forest (principal modelo)
            rf_model = joblib.load(models_dir / f'random_forest_{dataset_name}.pkl')
            y_pred = rf_model.predict(X_test)
            y_proba = rf_model.predict_proba(X_test)
            
            # Calcular métricas
            accuracy = (y_pred == y_test).mean()
            errors = (y_pred != y_test).sum()
            confidence = np.max(y_proba, axis=1)
            
            print(f"✓ Acurácia: {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"✓ Erros: {errors}/{len(y_test)}")
            print(f"✓ Confiança média: {confidence.mean():.3f}")
            
            # Matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            print(f"✓ Matriz confusão: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
            
            # Verificar se é overfitting
            if errors == 0:
                print("⚠️  ZERO ERROS - Possível overfitting!")
                all_perfect = False
            
            if confidence.min() > 0.95:
                print("⚠️  Confiança muito alta - Dados podem ser muito simples")
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("📋 ANÁLISE GERAL:")
    
    if all_perfect:
        print("🚨 TODOS OS MODELOS TÊM 100% ACURÁCIA")
        print("📌 Isso pode indicar:")
        print("   • Dados sintéticos muito bem separados")
        print("   • Padrões óbvios entre ataques e tráfego normal")
        print("   • Possível overfitting")
        print("   • Para dados reais, espere 85-95% acurácia")
    else:
        print("✅ Modelos parecem realistas")
    
    print("\n💡 RECOMENDAÇÕES:")
    print("   • Para testes: 100% está OK (dados sintéticos)")
    print("   • Para produção: Use dados reais do CIC-DDoS2019")
    print("   • Considere adicionar ruído aos dados sintéticos")
    print("   • Teste com dados não vistos durante treino")
    
    return True

if __name__ == "__main__":
    validate_models()
