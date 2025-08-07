#!/bin/bash
#
# Build and deployment script for DDoS Detection & Mitigation Lab
# LOCATION: deployment/scripts/make.sh
#
# Usage: ./deployment/scripts/make.sh <command> [options]
#

set -euo pipefail

# Detect project directory (2 levels up)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
SCRIPTS_DIR="$PROJECT_ROOT/deployment/scripts"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Logging functions
log_header() {
    echo -e "\n${CYAN}${BOLD}=== $1 ===${NC}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verify correct project structure
check_project_structure() {
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        log_error "docker-compose.yml not found: $DOCKER_COMPOSE_FILE"
        log_info "Run script from correct directory: ./deployment/scripts/make.sh"
        exit 1
    fi
    
    # Check essential directories
    local required_dirs=("src/datasets" "src/ml-processor" "src/bgp-controller")
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$PROJECT_ROOT/$dir" ]; then
            log_warning "Directory $dir not found - some features may fail"
        fi
    done
}

# Verify Docker availability and functionality
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Run: ./setup/linux/setup-complete.sh"
        exit 1
    fi
    
    if ! docker info &> /dev/null 2>&1; then
        log_error "Docker daemon not running. Try:"
        echo "  sudo systemctl start docker"
        echo "  sudo usermod -aG docker \$USER && logout/login"
        exit 1
    fi
}

# Verify Docker Compose and return command
check_docker_compose() {
    local COMPOSE_CMD=""
    
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        log_error "Docker Compose not found. Run: ./setup/linux/setup-complete.sh"
        exit 1
    fi
    
    echo "$COMPOSE_CMD"
}

# Execute docker-compose in correct directory
run_compose() {
    local compose_cmd="$1"
    local args="$2"
    local quiet="${3:-false}"
    
    check_project_structure
    cd "$PROJECT_ROOT"
    
    # Filtrar linhas do tipo '#N ...' (progresso do Docker) e mostrar apenas erros/avisos
    FILTER_CMD="grep -v '^#'"
    if [ "$quiet" = "true" ]; then
        if $compose_cmd -f "$DOCKER_COMPOSE_FILE" $args 2>&1 | $FILTER_CMD; then
            echo -e "  ${GREEN}Containers built and started${NC}"
        else
            echo -e "  ${RED}Build failed - showing full output${NC}"
            log_error "Build failed, running with full output..."
            $compose_cmd -f "$DOCKER_COMPOSE_FILE" $args
        fi
    else
        log_info "Executing: $compose_cmd $args"
        $compose_cmd -f "$DOCKER_COMPOSE_FILE" $args 2>&1 | $FILTER_CMD
    fi
}

# Verify datasets availability
check_datasets() {
    local datasets_dir="$PROJECT_ROOT/src/datasets/integrated"
    
    if [ ! -d "$datasets_dir" ] || [ ! -f "$datasets_dir/X_integrated_real.npy" ]; then
        log_warning "Datasets not found in $datasets_dir"
        log_info "Generating test data..."
        
        # Generate simple test data if not available
        python3 -c "
import numpy as np
import os

os.makedirs('$datasets_dir', exist_ok=True)
np.random.seed(42)
X = np.random.randn(1000, 20)
y = (X[:, 0] + X[:, 1]) > 0

np.save('$datasets_dir/X_integrated_real.npy', X)
np.save('$datasets_dir/y_integrated_real.npy', y.astype(int))

print('Test data generated')
" || log_error "Failed to generate test data"
    fi
}

# Help function
show_help() {
    log_header "DDoS Detection & Mitigation Lab - Commands"
    echo ""
    echo -e "${YELLOW}Usage: ./deployment/scripts/make.sh <command> [options]${NC}"
    echo ""
    echo -e "${GREEN}Main Commands:${NC}"
    echo "  up, start             Start all services"
    echo "  down, stop            Stop all services"
    echo "  restart               Restart all services"
    echo "  status, ps            Show container status"
    echo "  logs                  Show real-time logs"
    echo "  clean, cleanup        Clean containers and volumes"
    echo "  rebuild               Rebuild and restart"
    echo "  test                  Test basic configuration"
    echo "  shell [service]       Connect to service shell"
    echo ""
    echo -e "${CYAN}Machine Learning Commands:${NC}"
    echo "  train                 Train models (hybrid advanced)"
    echo "  train-simple          Train basic model" 
    echo "  train-lite            Train lite model (minimal memory)"
    echo "  train-advanced        Train with feature engineering"
    echo "  train-clean           Train with clean datasets (recommended)"
    echo "  create-clean          Create clean optimized datasets"
    echo "  create-realistic      Create realistic datasets with noise and overlap"
    echo "  train-realistic       Train with realistic datasets (99.x% accuracy)"
    echo "  validate              Validate model results and check for overfitting"
    echo "  optimize              Optimize hyperparameters"
    echo "  benchmark             Run complete benchmark"
    echo "  analyze               Analyze results"
    echo "  demo                  Run interactive model demonstration"
    echo ""
    echo "  help                  Show this help"
    echo ""
    echo -e "${YELLOW}Web Interfaces (after startup):${NC}"
    echo "  Grafana:    http://localhost:3000 (admin/admin123)"
    echo "  Prometheus: http://localhost:9090"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  ./deployment/scripts/make.sh up"
    echo "  ./deployment/scripts/make.sh train"
    echo "  ./deployment/scripts/make.sh logs"
    echo "  ./deployment/scripts/make.sh shell ml-processor"
    echo ""
    echo -e "${BLUE}First time setup:${NC}"
    echo "  1. Run: ./setup/linux/setup-complete.sh"
    echo "  2. Logout/login"
    echo "  3. Run: ./deployment/scripts/make.sh up"
}

# Função principal
main() {
    local command="${1:-help}"
    
    case "$command" in
        "up"|"start")
            log_header "Starting DDoS Detection System"
            check_docker
            check_datasets
            local COMPOSE_CMD=$(check_docker_compose)
            
            log_info "Building images and starting containers..."
            
            # Check if being executed by first-run.sh
            if [ "${DDOS_FIRST_RUN:-false}" = "true" ]; then
                run_compose "$COMPOSE_CMD" "up -d --build" "true"
            else
                run_compose "$COMPOSE_CMD" "up -d --build"
            fi
            
            log_success "Containers started successfully"
            log_info "Waiting for service initialization..."
            
            # Wait for services to be ready
            sleep 5
            
            # Show container status
            echo ""
            log_info "Container Status:"
            $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" | while IFS= read -r line; do
                if [[ $line == *"NAME"* ]]; then
                    echo -e "  ${CYAN}$line${NC}"
                elif [[ $line == *"Up"* ]]; then
                    echo -e "  ${GREEN}$line${NC}"
                else
                    echo -e "  ${YELLOW}$line${NC}"
                fi
            done
            
            sleep 5
            
            echo ""
            log_success "DDoS Detection System ready"
            echo ""
            log_info "Web interfaces:"
            log_info "- Grafana: http://localhost:3000 (admin/admin123)"
            log_info "- Prometheus: http://localhost:9090"
            echo ""
            log_info "Useful commands:"
            log_info "- Status: ./deployment/scripts/make.sh status"
            log_info "- Logs: ./deployment/scripts/make.sh logs"
            log_info "- Stop: ./deployment/scripts/make.sh down"
            ;;
        "down"|"stop")
            log_header "Stopping DDoS Detection System"
            local COMPOSE_CMD=$(check_docker_compose)
            run_compose "$COMPOSE_CMD" "down"
            log_success "System stopped successfully"
            ;;
        "restart")
            log_header "Restarting DDoS Detection System"
            local COMPOSE_CMD=$(check_docker_compose)
            log_info "Stopping containers..."
            run_compose "$COMPOSE_CMD" "down"
            log_info "Starting containers..."
            run_compose "$COMPOSE_CMD" "up -d"
            log_success "System restarted successfully"
            ;;
        "status"|"ps")
            log_header "DDoS Detection System Status"
            local COMPOSE_CMD=$(check_docker_compose)
            run_compose "$COMPOSE_CMD" "ps"
            echo ""
            log_info "Checking service connectivity..."
            
            # Verificar se serviços estão respondendo
            local services=("http://localhost:3000" "http://localhost:9090")
            for service in "${services[@]}"; do
                if curl -s "$service" > /dev/null 2>&1; then
                    log_success "$service está respondendo"
                else
                    log_warning "$service não está respondendo"
                fi
            done
            ;;
        "logs")
            log_header "Logs do Sistema DDoS"
            log_info "Pressione Ctrl+C para sair dos logs"
            sleep 2
            local COMPOSE_CMD=$(check_docker_compose)
            run_compose "$COMPOSE_CMD" "logs -f --tail=100"
            ;;
        "clean"|"cleanup")
            log_header "Limpando Sistema DDoS"
            read -p "Isso irá remover todos os containers e volumes. Continuar? [y/N]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                local COMPOSE_CMD=$(check_docker_compose)
                run_compose "$COMPOSE_CMD" "down -v --remove-orphans"
                
                # Limpeza adicional
                log_info "Removendo imagens órfãs..."
                docker image prune -f > /dev/null 2>&1 || true
                
                log_success "Sistema limpo com sucesso!"
            else
                log_info "Limpeza cancelada."
            fi
            ;;
        "rebuild")
            log_header "Reconstruindo Sistema DDoS"
            local COMPOSE_CMD=$(check_docker_compose)
            run_compose "$COMPOSE_CMD" "down"
            run_compose "$COMPOSE_CMD" "build --no-cache"
            run_compose "$COMPOSE_CMD" "up -d"
            log_success "Sistema reconstruído com sucesso!"
            ;;
        "shell")
            local service="${2:-ml-processor}"
            log_header "Conectando ao shell do serviço: $service"
            local COMPOSE_CMD=$(check_docker_compose)
            run_compose "$COMPOSE_CMD" "exec $service /bin/bash"
            ;;
        "train")
            log_header "Treinando Modelo Híbrido Avançado"
            check_docker
            check_datasets
            log_info "Executando treino com feature engineering avançado..."
            
            cd "$PROJECT_ROOT"
            if [ -f "$SCRIPTS_DIR/train_hybrid_advanced.py" ]; then
                # Verificar recursos disponíveis
                log_info "Verificando recursos do sistema..."
                echo "RAM disponível: $(free -h | grep Mem | awk '{print $7}')"
                echo "Espaço em disco: $(df -h . | tail -1 | awk '{print $4}')"
                
                # Executar com timeout e tratamento de erro
                log_info "Iniciando treino (pode demorar alguns minutos)..."
                timeout 1800 python3 "$SCRIPTS_DIR/train_hybrid_advanced.py" || {
                    local exit_code=$?
                    if [ $exit_code -eq 124 ]; then
                        log_error "Treino interrompido por timeout (30 minutos)"
                        log_info "Sugestão: Use 'ddos-train-simple' para teste rápido"
                    elif [ $exit_code -eq 137 ]; then
                        log_error "Treino interrompido por falta de memória"
                        log_info "Sugestões:"
                        log_info "  - Use: ./deployment/scripts/make.sh train-lite"
                        log_info "  - Feche outros programas para liberar RAM"
                        log_info "  - Verifique se tem pelo menos 4GB RAM livres"
                    else
                        log_error "Treino falhou com código de saída: $exit_code"
                        log_info "Verifique os logs para mais detalhes"
                    fi
                    return 1
                }
                log_success "Treino concluído com sucesso!"
            else
                log_error "Script de treino não encontrado: $SCRIPTS_DIR/train_hybrid_advanced.py"
                exit 1
            fi
            ;;
        "train-simple")
            log_header "Treinando Modelo Básico"
            check_docker
            check_datasets
            log_info "Executando treino básico (rápido, baixo consumo de RAM)..."
            
            cd "$PROJECT_ROOT"
            log_info "Iniciando treino básico..."
            python3 -c "
print('Importando bibliotecas...')
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import pickle
from pathlib import Path

print('Verificando dados...')
# Procurar dados integrados
data_paths = [
    './src/datasets/integrated/X_integrated_real.npy',
    './src/datasets/integrated/X_integrated_simple.npy'
]

X, y = None, None
for path in data_paths:
    if os.path.exists(path):
        X = np.load(path)
        y = np.load(path.replace('X_', 'y_'))
        print(f'Dados carregados: {X.shape}')
        break

if X is None:
    print('Nenhum dado encontrado. Execute setup_datasets.sh primeiro')
    exit(1)

print('Treinando modelo básico...')
# Usar apenas uma amostra se dados muito grandes
if len(X) > 10000:
    indices = np.random.choice(len(X), 10000, replace=False)
    X = X[indices]
    y = y[indices]
    print(f'Usando amostra de {len(X)} exemplos para treino rápido')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo simples
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1-Score: {f1:.3f}')

# Salvar modelo
os.makedirs('./src/models/simple', exist_ok=True)
with open('./src/models/simple/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print('Modelo básico treinado e salvo em ./src/models/simple/')
" || log_error "Falha no treino básico"
            ;;
        "train-lite")
            log_header "Treinando Modelo Lite (Memória Mínima)"
            check_docker
            log_info "Executando treino lite para sistemas com pouca memória..."
            
            cd "$PROJECT_ROOT"
            if [ -f "$SCRIPTS_DIR/train_simple_lite.py" ]; then
                log_info "Iniciando treino lite (usa <1GB RAM)..."
                python3 "$SCRIPTS_DIR/train_simple_lite.py" || {
                    local exit_code=$?
                    log_error "Treino lite falhou com código: $exit_code"
                    log_info "Este é o modo mais básico - se falhou, verifique:"
                    log_info "  - Python 3 instalado corretamente"
                    log_info "  - Bibliotecas: pip install scikit-learn numpy pandas"
                    return 1
                }
                log_success "Modelo lite treinado com sucesso!"
                log_info "Modelo salvo em: ./src/models/simple/rf_model_lite.pkl"
            else
                log_error "Script train_simple_lite.py não encontrado"
                exit 1
            fi
            ;;
        "train-advanced")
            log_header "Treinando com Feature Engineering Avançado"
            check_docker
            check_datasets
            log_info "Executando feature engineering avançado e treino..."
            
            cd "$PROJECT_ROOT"
            if [ -f "$SCRIPTS_DIR/real_unsw_engineering.py" ]; then
                python3 "$SCRIPTS_DIR/real_unsw_engineering.py"
                if [ -f "$SCRIPTS_DIR/train_hybrid_advanced.py" ]; then
                    python3 "$SCRIPTS_DIR/train_hybrid_advanced.py"
                fi
            else
                log_warning "Script de feature engineering não encontrado"
                log_info "Executando treino avançado diretamente..."
                python3 "$SCRIPTS_DIR/train_hybrid_advanced.py"
            fi
            ;;
        "create-clean")
            log_header "Criando Datasets Limpos"
            log_info "Gerando datasets otimizados e limpos..."
            
            cd "$PROJECT_ROOT"
            if [ -f "$SCRIPTS_DIR/create_clean_datasets.py" ]; then
                python3 "$SCRIPTS_DIR/create_clean_datasets.py" || {
                    log_error "Falha na criação dos datasets limpos"
                    exit 1
                }
                log_success "Datasets limpos criados com sucesso"
            else
                log_error "Script create_clean_datasets.py não encontrado"
                exit 1
            fi
            ;;
        "create-realistic")
            log_header "Criando Datasets Realistas"
            log_info "Gerando datasets com ruído e overlap para treino mais desafiador..."
            
            cd "$PROJECT_ROOT"
            if [ -f "$SCRIPTS_DIR/create_realistic_datasets_v2.py" ]; then
                python3 "$SCRIPTS_DIR/create_realistic_datasets_v2.py" || {
                    log_error "Falha na criação dos datasets realistas"
                    exit 1
                }
                log_success "Datasets realistas criados com sucesso"
                log_info "Use 'train-realistic' para treinar com os novos datasets"
            else
                log_error "Script create_realistic_datasets_v2.py não encontrado"
                exit 1
            fi
            ;;
        "train-realistic")
            log_header "Treinando com Datasets Realistas"
            log_info "Utilizando datasets com overlap para accuracy ~99.x%..."
            
            # Verificar se datasets realistas existem
            realistic_dir="$PROJECT_ROOT/src/datasets/realistic"
            if [ ! -d "$realistic_dir" ]; then
                log_warning "Datasets realistas não encontrados. Criando automaticamente..."
                cd "$PROJECT_ROOT"
                python3 "$SCRIPTS_DIR/create_realistic_datasets_v2.py"
            fi
            
            cd "$PROJECT_ROOT"
            if [ -f "$SCRIPTS_DIR/train_realistic_datasets.py" ]; then
                python3 "$SCRIPTS_DIR/train_realistic_datasets.py" || {
                    log_error "Falha no treino com datasets realistas"
                    exit 1
                }
                log_success "Treino com datasets realistas concluído"
                log_info "Accuracy esperada: 99.0% - 99.8% (não 100%)"
            else
                log_error "Script train_realistic_datasets.py não encontrado"
                exit 1
            fi
            ;;
        "train-clean")
            log_header "Treinando com Datasets Limpos"
            log_info "Utilizando datasets otimizados para treino rápido..."
            
            # Verificar se datasets limpos existem
            clean_dir="$PROJECT_ROOT/src/datasets/clean"
            if [ ! -d "$clean_dir" ]; then
                log_warning "Datasets limpos não encontrados. Criando automaticamente..."
                cd "$PROJECT_ROOT"
                python3 "$SCRIPTS_DIR/create_clean_datasets.py"
            fi
            
            cd "$PROJECT_ROOT"
            if [ -f "$SCRIPTS_DIR/train_clean_datasets.py" ]; then
                python3 "$SCRIPTS_DIR/train_clean_datasets.py" || {
                    log_error "Falha no treino com datasets limpos"
                    exit 1
                }
                log_success "Treino com datasets limpos concluído"
            else
                log_error "Script train_clean_datasets.py não encontrado"
                exit 1
            fi
            ;;
        "validate")
            log_header "Validando Qualidade dos Modelos"
            log_info "Verificando se resultados são realistas (não overfitting)..."
            
            cd "$PROJECT_ROOT"
            
            # Verificar se existem modelos realistas
            if [ -d "$PROJECT_ROOT/src/models/realistic" ]; then
                log_info "Validando modelos realistas..."
                if [ -f "$SCRIPTS_DIR/validate_realistic_models.py" ]; then
                    python3 "$SCRIPTS_DIR/validate_realistic_models.py" || {
                        log_error "Falha na validação dos modelos realistas"
                        exit 1
                    }
                    log_success "Validação de modelos realistas concluída"
                else
                    log_warning "Script validate_realistic_models.py não encontrado"
                fi
            fi
            
            # Validação dos modelos antigos se existirem
            if [ -f "$SCRIPTS_DIR/validate_models.py" ]; then
                log_info "Validando modelos antigos também..."
                python3 "$SCRIPTS_DIR/validate_models.py" || {
                    log_warning "Falha na validação dos modelos antigos (não crítico)"
                }
            else
                log_info "Script validate_models.py não encontrado (normal)"
            fi
            ;;
        "optimize")
            log_header "Otimizando Hiperparâmetros"
            check_docker
            check_datasets
            log_info "Executando otimização de hiperparâmetros..."
            
            cd "$PROJECT_ROOT"
            for script in "optimize_xgboost.py" "optimize_isolation_forest.py"; do
                if [ -f "$SCRIPTS_DIR/$script" ]; then
                    log_info "Executando $script..."
                    python3 "$SCRIPTS_DIR/$script"
                fi
            done
            ;;
        "benchmark")
            log_header "Executando Benchmark Completo"
            check_docker
            check_datasets
            log_info "Executando benchmark de performance..."
            
            cd "$PROJECT_ROOT"
            if [ -f "$SCRIPTS_DIR/benchmark.py" ]; then
                python3 "$SCRIPTS_DIR/benchmark.py"
            else
                log_error "Script de benchmark não encontrado"
                exit 1
            fi
            ;;
        "analyze")
            log_header "Analisando Resultados"
            check_docker
            log_info "Analisando resultados dos modelos..."
            
            cd "$PROJECT_ROOT"
            if [ -f "$SCRIPTS_DIR/analyze_results.py" ]; then
                python3 "$SCRIPTS_DIR/analyze_results.py"
            else
                log_error "Script de análise não encontrado"
                exit 1
            fi
            ;;
        "demo")
            log_header "Demonstração Interativa do Modelo"
            check_datasets
            log_info "Executando demonstração do modelo DDoS..."
            
            cd "$PROJECT_ROOT"
            if [ -f "$SCRIPTS_DIR/demo_modelo_ddos.py" ]; then
                python3 "$SCRIPTS_DIR/demo_modelo_ddos.py"
            else
                log_error "Script de demo não encontrado: $SCRIPTS_DIR/demo_modelo_ddos.py"
                exit 1
            fi
            ;;
        "test")
            log_header "Executando Testes do Sistema"
            check_datasets
            log_info "Testando geração de dados..."
            python3 -c "
import os
import numpy as np
print('NumPy funcionando')
import pandas as pd  
print('Pandas funcionando')
data_path = '$PROJECT_ROOT/src/datasets/integrated/X_integrated_real.npy'
if os.path.exists(data_path):
    X = np.load(data_path)
    print(f'Dados carregados: {X.shape}')
else:
    print('Dados não encontrados')
" || log_error "Teste falhou"
            ;;
        "help"|"")
            show_help
            ;;
        *)
            log_error "Comando desconhecido: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Verificar se o script está sendo executado diretamente
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
