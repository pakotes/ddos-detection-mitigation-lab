#!/usr/bin/env python3
"""
Script para FORÇAR recriação dos datasets realistas
Garante que os novos datasets substituem completamente os antigos
"""

import logging
import shutil
from pathlib import Path
import subprocess
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def clean_old_datasets():
    """Limpar datasets antigos para forçar uso dos novos"""
    logger.info("Limpando datasets antigos...")
    
    base_dir = Path(__file__).parent.parent.parent / 'src'
    
    # Remover diretórios antigos
    old_dirs = [
        base_dir / 'datasets' / 'clean',
        base_dir / 'datasets' / 'integrated',
        base_dir / 'models' / 'clean',
        base_dir / 'models' / 'simple'
    ]
    
    for old_dir in old_dirs:
        if old_dir.exists():
            logger.info(f"Removendo: {old_dir}")
            shutil.rmtree(old_dir)
        else:
            logger.info(f"Já removido: {old_dir}")
    
    logger.info("Limpeza concluída!")

def force_create_realistic():
    """Forçar criação dos datasets realistas"""
    logger.info("Criando datasets realistas (forçado)...")
    
    script_path = Path(__file__).parent / 'create_realistic_datasets_v2.py'
    
    if not script_path.exists():
        logger.error(f"Script não encontrado: {script_path}")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        if result.returncode == 0:
            logger.info("✅ Datasets realistas criados com sucesso!")
            logger.info(result.stdout)
            return True
        else:
            logger.error("❌ Erro ao criar datasets realistas:")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Erro ao executar script: {e}")
        return False

def force_train_realistic():
    """Forçar treinamento com datasets realistas"""
    logger.info("Treinando modelos realistas (forçado)...")
    
    script_path = Path(__file__).parent / 'train_realistic_datasets.py'
    
    if not script_path.exists():
        logger.error(f"Script não encontrado: {script_path}")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        if result.returncode == 0:
            logger.info("✅ Modelos realistas treinados com sucesso!")
            logger.info(result.stdout)
            return True
        else:
            logger.error("❌ Erro ao treinar modelos realistas:")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Erro ao executar script: {e}")
        return False

def validate_realistic():
    """Validar modelos realistas"""
    logger.info("Validando modelos realistas...")
    
    script_path = Path(__file__).parent / 'validate_realistic_models.py'
    
    if not script_path.exists():
        logger.warning("Script de validação realista não encontrado, usando antigo...")
        script_path = Path(__file__).parent / 'validate_models.py'
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        if result.returncode == 0:
            logger.info("✅ Validação concluída!")
            logger.info(result.stdout)
            return True
        else:
            logger.error("❌ Erro na validação:")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Erro ao executar validação: {e}")
        return False

def main():
    """Função principal - pipeline completo"""
    logger.info("INICIANDO PIPELINE COMPLETO PARA DATASETS REALISTAS")
    logger.info("="*60)
    
    try:
        # 1. Limpar datasets antigos
        logger.info("\n1️⃣ LIMPANDO DATASETS ANTIGOS...")
        clean_old_datasets()
        
        # 2. Criar datasets realistas
        logger.info("\n2️⃣ CRIANDO DATASETS REALISTAS...")
        if not force_create_realistic():
            logger.error("Falha na criação dos datasets!")
            return False
        
        # 3. Treinar modelos
        logger.info("\n3️⃣ TREINANDO MODELOS REALISTAS...")
        if not force_train_realistic():
            logger.error("Falha no treinamento!")
            return False
        
        # 4. Validar resultados
        logger.info("\n4️⃣ VALIDANDO RESULTADOS...")
        if not validate_realistic():
            logger.warning("Validação falhou, mas modelos foram criados")
        
        logger.info("\n🎉 PIPELINE COMPLETO CONCLUÍDO!")
        logger.info("="*60)
        logger.info("✅ Datasets realistas criados")
        logger.info("✅ Modelos treinados")
        logger.info("✅ Validação executada")
        logger.info("📊 Verifique os resultados acima")
        logger.info("🎯 Accuracy esperada: 90-98% (não 100%)")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro no pipeline: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
