#!/usr/bin/env python3
"""
Pipeline de Preparação de Conjuntos de Dados

Este módulo orquestra o processo completo de preparação de conjuntos de dados 
para o DDoS Mitigation Lab. Coordena o processamento de conjuntos de dados 
individuais e a sua integração em múltiplas configurações de treino.

"""

import sys
import subprocess
from pathlib import Path
import logging
import json

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetPreparationPipeline:
    """
    Controlador principal do pipeline de preparação de conjuntos de dados.
    
    Esta classe gere a execução de processadores individuais de conjuntos de dados
    e a integração de conjuntos de dados processados em várias configurações de
    treino adequadas para diferentes abordagens de aprendizagem automática.
    """
    
    def __init__(self):
        """Initialize the preparation pipeline"""
        self.script_dir = Path(__file__).parent
        self.base_dir = self.script_dir.parent.parent
        self.datasets_dir = self.base_dir / "src" / "datasets"
        self.processed_dir = self.datasets_dir / "processed"
        self.integrated_dir = self.processed_dir / "integrated"
        
        # Ensure output directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.integrated_dir.mkdir(parents=True, exist_ok=True)
        
        # Define processing scripts
        self.processors = {
            'nf_unsw': self.script_dir / "process_nf_unsw.py",
            'cic_ddos': self.script_dir / "process_cic_ddos.py",
            'integration': self.script_dir / "integrate_datasets.py"
        }
    
    def check_dataset_availability(self):
        """Verificar que conjuntos de dados estão disponíveis para processamento"""
        available_datasets = {}
        
        # Check for NF-UNSW-NB15-v3 (local directory)
        nf_unsw_dir = self.script_dir / "NF-UNSW-NB15-v3"
        if nf_unsw_dir.exists() and list(nf_unsw_dir.glob("*.csv")):
            available_datasets['nf_unsw'] = {
                'path': nf_unsw_dir,
                'files': len(list(nf_unsw_dir.glob("*.csv"))),
                'description': 'Conjunto de dados NetFlow NF-UNSW-NB15-v3 para detecção geral de intrusões'
            }
        
        # Check for CIC-DDoS2019 (local directory)
        cic_ddos_dir = self.script_dir / "CIC-DDoS2019"
        if cic_ddos_dir.exists():
            csv_files = []
            for subdir in cic_ddos_dir.iterdir():
                if subdir.is_dir():
                    csv_files.extend(list(subdir.glob("*.csv")))
            csv_files.extend(list(cic_ddos_dir.glob("*.csv")))
            
            if csv_files:
                available_datasets['cic_ddos'] = {
                    'path': cic_ddos_dir,
                    'files': len(csv_files),
                    'description': 'Conjunto de dados CIC-DDoS2019 para detecção especializada de DDoS'
                }
        
        return available_datasets
    
    def run_processor(self, processor_name, script_path):
        """Executar um script de processamento de conjunto de dados, mostrando o progresso em tempo real."""
        logger.info(f"A iniciar processador {processor_name}")
        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=self.script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            # Mostrar a saída em tempo real
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line, end='')
            returncode = process.wait()
            if returncode == 0:
                logger.info(f"Processador {processor_name} completado com sucesso")
                return True
            else:
                logger.error(f"Processador {processor_name} falhou (return code {returncode})")
                return False
        except Exception as e:
            logger.error(f"Error running {processor_name} processor: {str(e)}")
            return False
    
    def verify_processed_output(self, processor_name):
        """Verify that a processor generated the expected output files"""
        expected_files = [
            f"X_{processor_name}.npy",
            f"y_{processor_name}.npy",
            f"metadata_{processor_name}.json"
        ]
        
        for filename in expected_files:
            file_path = self.processed_dir / filename
            if not file_path.exists():
                logger.warning(f"Expected output file missing: {filename}")
                return False
        
        logger.info(f"Verified output files for {processor_name}")
        return True
    
    def generate_processing_report(self, results, available_datasets):
        """Gerar um relatório abrangente do processamento"""
        report = {
            'processing_date': str(Path().resolve()),
            'pipeline_version': '1.0',
            'available_datasets': available_datasets,
            'processing_results': results,
            'output_directory': str(self.processed_dir),
            'integration_directory': str(self.integrated_dir)
        }
        
        # Count successful processors
        successful_processors = sum(1 for result in results.values() if result.get('success', False))
        
        report['summary'] = {
            'datasets_available': len(available_datasets),
            'processors_run': len(results),
            'processors_successful': successful_processors,
            'integration_attempted': 'integration' in results,
            'integration_successful': results.get('integration', {}).get('success', False)
        }
        
        # Save report
        report_file = self.processed_dir / "processing_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing report saved to: {report_file}")
        return report
    
    def execute_pipeline(self):
        """Execute the complete dataset preparation pipeline"""
        logger.info("Starting dataset preparation pipeline")
        print("Pipeline de Preparação de Conjuntos de Dados")
        print("=" * 60)
        
        # Check dataset availability
        available_datasets = self.check_dataset_availability()
        
        if not available_datasets:
            print("Nenhum conjunto de dados encontrado para processamento.")
            print("Por favor, certifique-se de que os conjuntos de dados estão colocados nos diretórios corretos:")
            print(f"  - NF-UNSW-NB15-v3: {self.script_dir / 'NF-UNSW-NB15-v3'}")
            print(f"  - CIC-DDoS2019: {self.script_dir / 'CIC-DDoS2019'}")
            return False
        
        print(f"Encontrados {len(available_datasets)} conjunto(s) de dados para processamento:")
        for name, info in available_datasets.items():
            print(f"  - {name}: {info['files']} ficheiros - {info['description']}")
        
        print()
        
        # Execute individual processors
        processing_results = {}
        
        for dataset_name in available_datasets.keys():
            processor_script = self.processors.get(dataset_name)
            
            if processor_script and processor_script.exists():
                print(f"A processar {dataset_name}...")
                
                success = self.run_processor(dataset_name, processor_script)
                
                if success:
                    # Verify output files were created
                    verified = self.verify_processed_output(dataset_name)
                    processing_results[dataset_name] = {
                        'success': verified,
                        'output_verified': verified,
                        'script': str(processor_script)
                    }
                    
                    if verified:
                        print(f"  Processamento de {dataset_name} completado com sucesso")
                    else:
                        print(f"  Processamento de {dataset_name} completado mas a verificação da saída falhou")
                else:
                    processing_results[dataset_name] = {
                        'success': False,
                        'output_verified': False,
                        'script': str(processor_script)
                    }
                    print(f"  Processamento de {dataset_name} falhou")
            else:
                print(f"  Aviso: Nenhum script de processamento encontrado para {dataset_name}")
                processing_results[dataset_name] = {
                    'success': False,
                    'output_verified': False,
                    'error': 'No processor script found'
                }
        
        # Execute integration if we have processed datasets
        successful_processors = [name for name, result in processing_results.items() 
                               if result.get('success', False)]
        
        if successful_processors:
            print(f"\nA integrar {len(successful_processors)} conjunto(s) de dados processado(s)...")
            
            integration_script = self.processors['integration']
            if integration_script.exists():
                integration_success = self.run_processor('integration', integration_script)
                processing_results['integration'] = {
                    'success': integration_success,
                    'processed_datasets': successful_processors,
                    'script': str(integration_script)
                }
                
                if integration_success:
                    print("  Integração de conjuntos de dados completada com sucesso")
                else:
                    print("  Integração de conjuntos de dados falhou")
            else:
                print("  Aviso: Script de integração não encontrado")
        else:
            print("\nA saltar integração - nenhum conjunto de dados processado com sucesso")
        
        # Generate final report
        report = self.generate_processing_report(processing_results, available_datasets)
        
        # Print summary
        print("\n" + "=" * 60)
        print("RESUMO DO PROCESSAMENTO")
        print("=" * 60)
        print(f"Conjuntos de dados disponíveis: {report['summary']['datasets_available']}")
        print(f"Processadores executados: {report['summary']['processors_run']}")
        print(f"Processadores bem-sucedidos: {report['summary']['processors_successful']}")
        
        if report['summary']['integration_attempted']:
            integration_status = "bem-sucedida" if report['summary']['integration_successful'] else "falhada"
            print(f"Integração: {integration_status}")
        
        print(f"Diretório de saída: {self.processed_dir}")
        
        if report['summary']['processors_successful'] > 0:
            print("\nProcessamento completado com sucesso!")
            return True
        else:
            print("\nProcessamento falhado - nenhum conjunto de dados foi processado com sucesso.")
            return False

def main():
    """Ponto de entrada principal para o pipeline de preparação de conjuntos de dados"""
    try:
        pipeline = DatasetPreparationPipeline()
        success = pipeline.execute_pipeline()
        
        if success:
            logger.info("Pipeline de preparação de conjuntos de dados completado com sucesso")
            return 0
        else:
            logger.error("Pipeline de preparação de conjuntos de dados falhado")
            return 1
            
    except KeyboardInterrupt:
        print("\nProcessamento interrompido pelo utilizador")
        logger.info("Processamento interrompido pelo utilizador")
        return 1
    except Exception as e:
        print(f"\nErro inesperado: {str(e)}")
        logger.error(f"Erro inesperado no pipeline: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
