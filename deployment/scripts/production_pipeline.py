#!/usr/bin/env python3
"""
Pipeline completo de produção para DDoS Detection
Executa todo o processo: dados reais → modelo otimizado → validação → produção
"""

import logging
import sys
import time
from pathlib import Path
import subprocess
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

class ProductionPipeline:
    """Pipeline completo para modelo DDoS de produção"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.scripts_dir = self.project_root / 'deployment' / 'scripts'
        self.models_dir = self.project_root / 'src' / 'models'
        
    def log_step(self, step_name, description):
        """Log formatado para cada etapa"""
        logger.info(f"\n{'='*60}")
        logger.info(f"ETAPA {step_name}: {description}")
        logger.info(f"{'='*60}")
    
    def run_script(self, script_name, description):
        """Executar script Python e verificar resultado"""
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            logger.error(f"Script não encontrado: {script_path}")
            return False
        
        logger.info(f"Executando: {script_name}")
        logger.info(f"Descrição: {description}")
        
        try:
            # Executar script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutos timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {script_name} executado com sucesso")
                if result.stdout:
                    # Mostrar apenas últimas linhas importantes
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-5:]:
                        if line.strip():
                            logger.info(f"  {line}")
                return True
            else:
                logger.error(f"❌ Falha em {script_name}")
                if result.stderr:
                    logger.error(f"Erro: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout em {script_name} (30 min)")
            return False
        except Exception as e:
            logger.error(f"❌ Erro executando {script_name}: {e}")
            return False
    
    def check_data_availability(self):
        """Verificar se dados estão disponíveis"""
        self.log_step("0", "Verificação de Dados")
        
        datasets_dir = self.project_root / 'src' / 'datasets'
        
        # Verificar datasets integrados
        integrated_dir = datasets_dir / 'integrated'
        if integrated_dir.exists():
            files = list(integrated_dir.glob('*.npy'))
            if files:
                logger.info(f"✅ Dados integrados encontrados: {len(files)} arquivos")
                return True
        
        # Verificar dados reais
        real_dir = datasets_dir / 'real'
        if real_dir.exists():
            files = list(real_dir.glob('*.npy'))
            if files:
                logger.info(f"✅ Dados reais encontrados: {len(files)} arquivos")
                return True
        
        logger.warning("⚠️ Nenhum dado encontrado - será necessário processar")
        return False
    
    def process_real_data(self):
        """Processar dados reais se necessário"""
        self.log_step("1", "Processamento de Dados Reais")
        
        # Verificar se dados reais já existem
        real_dir = self.project_root / 'src' / 'datasets' / 'real'
        if real_dir.exists() and list(real_dir.glob('*.npy')):
            logger.info("✅ Dados reais já processados")
            return True
        
        return self.run_script(
            'process_real_datasets.py',
            'Processamento de CIC-DDoS2019 e UNSW-NB15 originais'
        )
    
    def train_models(self):
        """Treinar modelos com dados reais"""
        self.log_step("2", "Treinamento com Dados Reais")
        
        return self.run_script(
            'train_real_datasets.py',
            'Treinamento com datasets originais'
        )
    
    def validate_models(self):
        """Validar qualidade dos modelos"""
        self.log_step("3", "Validação de Modelos")
        
        # Primeiro tentar validação realista
        realistic_dir = self.models_dir / 'realistic'
        if realistic_dir.exists():
            success = self.run_script(
                'validate_realistic_models.py',
                'Validação de modelos realistas'
            )
            if success:
                return True
        
        # Fallback para validação padrão
        return self.run_script(
            'validate_models.py',
            'Validação de modelos padrão'
        )
    
    def create_production_model(self):
        """Criar modelo otimizado para produção"""
        self.log_step("4", "Criação do Modelo de Produção")
        
        return self.run_script(
            'create_production_model.py',
            'Criação de modelo otimizado para produção'
        )
    
    def test_production_model(self):
        """Testar modelo de produção"""
        self.log_step("5", "Teste do Modelo de Produção")
        
        success = self.run_script(
            'test_production_model.py',
            'Teste de performance em condições de produção'
        )
        
        if success:
            # Tentar ler relatório de teste
            report_path = self.models_dir / 'production' / 'production_test_report.json'
            if report_path.exists():
                try:
                    with open(report_path) as f:
                        report = json.load(f)
                    
                    readiness = report.get('production_readiness', {})
                    ready = all(readiness.values())
                    
                    logger.info(f"\n📊 RELATÓRIO DE PRODUÇÃO:")
                    logger.info(f"  Pronto para produção: {'✅ SIM' if ready else '⚠️ NECESSITA AJUSTES'}")
                    
                    if 'accuracy_test' in report:
                        acc = report['accuracy_test']
                        logger.info(f"  Accuracy: {acc.get('accuracy', 0):.3f}")
                        logger.info(f"  Precision: {acc.get('precision', 0):.3f}")
                        logger.info(f"  Taxa FP: {acc.get('false_positive_rate', 0):.3f}")
                    
                    if 'speed_benchmark' in report:
                        speed = report['speed_benchmark']
                        logger.info(f"  Velocidade: {speed.get('samples_per_second', 0):,.0f} amostras/s")
                    
                except Exception as e:
                    logger.warning(f"Erro lendo relatório: {e}")
        
        return success
    
    def generate_final_report(self):
        """Gerar relatório final do pipeline"""
        self.log_step("6", "Relatório Final")
        
        # Verificar arquivos gerados
        production_dir = self.models_dir / 'production'
        model_file = production_dir / 'ddos_production_model.pkl'
        report_file = production_dir / 'production_test_report.json'
        
        logger.info("🎯 RESULTADOS DO PIPELINE:")
        
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            logger.info(f"  ✅ Modelo de produção: {model_file} ({size_mb:.1f} MB)")
        else:
            logger.error(f"  ❌ Modelo de produção não encontrado!")
        
        if report_file.exists():
            logger.info(f"  ✅ Relatório de teste: {report_file}")
        else:
            logger.warning(f"  ⚠️ Relatório de teste não encontrado")
        
        # Resumo de modelos disponíveis
        logger.info(f"\n📁 MODELOS DISPONÍVEIS:")
        for model_type in ['production', 'realistic', 'hybrid_advanced', 'hybrid']:
            model_dir = self.models_dir / model_type
            if model_dir.exists():
                model_files = list(model_dir.glob('*.pkl'))
                logger.info(f"  {model_type}: {len(model_files)} modelos")
        
        logger.info(f"\n🚀 COMANDOS ÚTEIS:")
        logger.info(f"  Testar modelo: python {self.scripts_dir}/test_production_model.py")
        logger.info(f"  Ver relatório: cat {report_file}")
        
        return True
    
    def run_complete_pipeline(self, skip_validation=False):
        """Executar pipeline completo"""
        logger.info("🏭 INICIANDO PIPELINE DE PRODUÇÃO DDoS")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Etapa 0: Verificar dados
            if not self.check_data_availability():
                logger.warning("Dados não encontrados - pipeline processará do zero")
            
            # Etapa 1: Processar dados reais
            if not self.process_real_data():
                logger.error("❌ Falha no processamento de dados")
                return False
            
            # Etapa 2: Treinar modelos
            if not self.train_models():
                logger.error("❌ Falha no treinamento")
                return False
            
            # Etapa 3: Validar (opcional)
            if not skip_validation:
                if not self.validate_models():
                    logger.warning("⚠️ Falha na validação - continuando")
            
            # Etapa 4: Criar modelo de produção
            if not self.create_production_model():
                logger.error("❌ Falha na criação do modelo de produção")
                return False
            
            # Etapa 5: Testar modelo de produção
            if not self.test_production_model():
                logger.error("❌ Falha no teste de produção")
                return False
            
            # Etapa 6: Relatório final
            self.generate_final_report()
            
            elapsed = time.time() - start_time
            logger.info(f"\n🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
            logger.info(f"⏱️ Tempo total: {elapsed/60:.1f} minutos")
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("❌ Pipeline interrompido pelo usuário")
            return False
        except Exception as e:
            logger.error(f"❌ Erro no pipeline: {e}")
            return False

def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline de produção DDoS')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Pular etapa de validação')
    parser.add_argument('--step', type=str, choices=[
        'process', 'train', 'validate', 'create', 'test', 'report'
    ], help='Executar apenas uma etapa específica')
    
    args = parser.parse_args()
    
    pipeline = ProductionPipeline()
    
    if args.step:
        # Executar etapa específica
        if args.step == 'process':
            success = pipeline.process_real_data()
        elif args.step == 'train':
            success = pipeline.train_models()
        elif args.step == 'validate':
            success = pipeline.validate_models()
        elif args.step == 'create':
            success = pipeline.create_production_model()
        elif args.step == 'test':
            success = pipeline.test_production_model()
        elif args.step == 'report':
            success = pipeline.generate_final_report()
    else:
        # Pipeline completo
        success = pipeline.run_complete_pipeline(args.skip_validation)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
