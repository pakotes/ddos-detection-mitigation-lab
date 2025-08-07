#!/bin/bash
#
# First-run setup script - DDoS Detection & Mitigation Lab
# Executes all necessary steps for initial setup
# LOCATION: first-run.sh
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Variables
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/ddos-first-run.log"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    # Substituir mensagem padrão se for a mensagem de sistema pronto
    if [[ "$1" == "DDoS Detection System ready" ]]; then
        echo -e "${GREEN}Sistema pronto${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "${GREEN}[OK]${NC} $1" | tee -a "$LOG_FILE"
    fi
}

log_warning() {
    echo -e "${YELLOW}[AVISO]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERRO]${NC} $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "\n${CYAN}${BOLD}=== $1 ===${NC}" | tee -a "$LOG_FILE"
}

print_banner() {
    clear
    echo -e "${BLUE}${BOLD}"
    cat << 'EOF'
================================================================
    Laboratório de Deteção e Mitigação de DDoS - Primeira Execução
================================================================
EOF
    echo -e "${NC}"
    echo "Este é o ponto de entrada recomendado para preparar e iniciar o laboratório."
    echo "• Valida se o sistema está pronto (Docker, Python, etc.)"
    echo "• Executa a configuração completa (install.sh) apenas se necessário"
    echo "• Cria atalhos úteis para facilitar o uso"
    echo "• Garante que os datasets estão prontos (gera sintéticos se faltar)"
    echo "• Inicia o sistema e mostra instruções de utilização."
    echo ""
}

check_system() {
    log_step "A verificar o sistema"
    
    local need_setup=false
    
    # Docker
    if ! command -v docker &> /dev/null; then
        log_warning "Docker não encontrado"
        need_setup=true
    elif ! docker info &> /dev/null 2>&1; then
        log_warning "Docker não está a funcionar"
        need_setup=true
    else
        log_success "Docker OK: $(docker --version)"
    fi

    # Docker Compose
    if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
        log_warning "Docker Compose não encontrado"
        need_setup=true
    else
        log_success "Docker Compose OK"
    fi

    # Python
    if ! command -v python3 &> /dev/null; then
        log_warning "Python3 não encontrado"
        need_setup=true
    else
        log_success "Python3 OK: $(python3 --version)"
    fi

    if [ "$need_setup" = true ]; then
        log_warning "O sistema precisa de configuração"
        return 1
    else
        log_success "O sistema já está configurado"
        return 0
    fi
}

run_setup() {
    log_step "A executar configuração completa"
    
    if [ -f "$PROJECT_ROOT/setup/install.sh" ]; then
        log_info "A preparar o ambiente base (install.sh)..."
        chmod +x "$PROJECT_ROOT/setup/install.sh"
        "$PROJECT_ROOT/setup/install.sh"
        # Validação pós-configuração
        log_info "A validar se o ambiente ficou corretamente configurado..."
        if ! check_system; then
            log_error "A configuração automática não conseguiu preparar o sistema corretamente. Por favor, consulte a documentação ou peça suporte."
            exit 1
        fi
    else
        log_error "Script de configuração não encontrado: $PROJECT_ROOT/setup/install.sh"
        return 1
    fi
}

create_aliases() {
    log_step "A criar atalhos convenientes"

    log_info "A criar atalhos diretamente..."
    
    # Criar aliases diretamente no ~/.bashrc
    local alias_file="$HOME/.ddos_aliases"
    cat > "$alias_file" << 'EOF'
# DDoS Lab Aliases
alias ddos-up='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh up'
alias ddos-down='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh down'
alias ddos-status='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh status'
alias ddos-logs='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh logs'
alias ddos-check='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh test'
alias ddos-rebuild='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh rebuild'

# Machine Learning Aliases
alias ddos-train='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh train'
alias ddos-train-simple='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh train-simple'
alias ddos-train-advanced='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh train-advanced'
alias ddos-train-clean='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh train-clean'
alias ddos-create-clean='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh create-clean'
alias ddos-create-realistic='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh create-realistic'
alias ddos-train-realistic='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh train-realistic'
alias ddos-validate='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh validate'
alias ddos-optimize='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh optimize'
alias ddos-benchmark='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh benchmark'
alias ddos-analyze='cd ~/ddos-detection-mitigation-lab && ./deployment/scripts/make.sh analyze'

# Demo e utilitários
alias ddos-demo='cd ~/ddos-detection-mitigation-lab && python deployment/scripts/demo_modelo_ddos.py'
alias ddos-cd='cd ~/ddos-detection-mitigation-lab'
EOF
    
    # Adicionar source ao bashrc se não existir
    if ! grep -q ".ddos_aliases" "$HOME/.bashrc" 2>/dev/null; then
        echo "source ~/.ddos_aliases" >> "$HOME/.bashrc"
    fi
    
    log_success "Atalhos criados em $alias_file"
}

check_datasets() {
    log_step "A verificar e preparar datasets"
    
    local datasets_dir="$PROJECT_ROOT/src/datasets/integrated"
    local clean_datasets_dir="$PROJECT_ROOT/src/datasets/clean"
    
    # Verificar se datasets reais existem
    if [ ! -f "$datasets_dir/X_integrated_real.npy" ] || [ ! -f "$datasets_dir/y_integrated_real.npy" ]; then
        log_warning "Datasets reais não encontrados. Serão gerados dados sintéticos de teste. Para uso real, coloque os datasets oficiais nas pastas indicadas em setup/dataset-preparation."
        generate_fallback_data
    else
        log_success "Datasets reais já existem"
        # Mostrar estatísticas
        python3 -c "
import numpy as np
try:
    X = np.load('$datasets_dir/X_integrated_real.npy')
    y = np.load('$datasets_dir/y_integrated_real.npy')
    print('Amostras disponíveis:', X.shape[0], 'com', X.shape[1], 'características')
    print('Total de ataques:', int(y.sum()), '(', round(100*y.mean(),1),'% )')
except Exception as e:
    print('Erro ao ler os datasets:', e)
" 2>/dev/null || log_info "Datasets encontrados (validação falhou)"
    fi

    # Criar datasets limpos otimizados
    if [ ! -d "$clean_datasets_dir" ] || [ ! -f "$clean_datasets_dir/summary.json" ]; then
        log_info "A criar datasets limpos otimizados..."

        # Verificar se script existe
        if [ -f "$PROJECT_ROOT/deployment/scripts/create_clean_datasets.py" ]; then
            cd "$PROJECT_ROOT"
            python3 "$PROJECT_ROOT/deployment/scripts/create_clean_datasets.py" || {
                log_warning "Falha na criação dos datasets limpos, mas a continuar..."
            }

            if [ -f "$clean_datasets_dir/summary.json" ]; then
                log_success "Datasets limpos otimizados criados"
                log_info "Disponíveis: CIC-DDoS2019 limpo, UNSW-NB15 limpo, Dataset integrado"
            else
                log_warning "Os datasets limpos não foram criados corretamente"
            fi
        else
            log_warning "Script de criação de datasets limpos não encontrado"
        fi
    else
        log_success "Datasets limpos já existem"
    fi
}

generate_fallback_data() {
datasets_dir = '$datasets_dir'
    log_info "A gerar dados de fallback para teste..."
    mkdir -p "$datasets_dir"

    python3 -c "
import numpy as np
np.save(f'{datasets_dir}/y_integrated_real.npy', y.astype(int))
    python3 -c "
import numpy as np
import json
import os

print('A gerar dados sintéticos para teste...')
np.random.seed(42)
X = np.random.randn(5000, 20)
y = (X[:, 0] + X[:, 1] + np.random.randn(5000) * 0.1) > 0
y = y.astype(int)
# Garantir pelo menos duas classes
if y.sum() == 0 or y.sum() == len(y):
    y[0] = 1 - y[0]

}
np.save(f'{datasets_dir}/X_integrated_real.npy', X)
np.save(f'{datasets_dir}/y_integrated_real.npy', y)

# Metadata
metadata = {
    'samples': len(X),
    'features': X.shape[1],
    'positive_ratio': float(y.mean()),
    'generated': 'synthetic_fallback_data',
    'note': 'Dados de fallback - substitua por datasets reais'
}

with open(f'{datasets_dir}/metadata_real.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'Dados de fallback gerados: {X.shape} amostras, {y.sum()} positivas')
print('IMPORTANTE: Estes são dados sintéticos para teste')
"
}

start_system() {
    log_step "A iniciar o sistema DDoS"
    
    cd "$PROJECT_ROOT"
    
    # Tornar make.sh executável
    chmod +x deployment/scripts/make.sh

    # Definir variável para modo silencioso
    export DDOS_FIRST_RUN=true

    # Iniciar sistema
    ./deployment/scripts/make.sh up

    log_success "Sistema iniciado"
}

show_instructions() {
    log_step "Sistema pronto!"

    echo ""
    echo -e "${GREEN}${BOLD}Laboratório de Deteção e Mitigação de DDoS em execução!${NC}"
    echo ""
    # Obter IP da máquina
    IP_MAQUINA=$(hostname -I | awk '{print $1}')
    echo -e "${CYAN}Dashboards disponíveis:${NC}"
    echo "   Grafana:    http://$IP_MAQUINA:3000 (admin/admin123)"
    echo "   Prometheus: http://$IP_MAQUINA:9090"
    echo ""
    echo -e "${CYAN}Comandos úteis:${NC}"
    echo "   ./deployment/scripts/make.sh status    # Ver estado"
    echo "   ./deployment/scripts/make.sh logs      # Ver registos"
    echo "   ./deployment/scripts/make.sh down      # Parar sistema"
    echo "   ./deployment/scripts/make.sh help      # Ajuda completa"
    echo ""
    if [ -f "$HOME/.ddos_aliases" ]; then
        echo -e "${CYAN}Atalhos disponíveis (após 'source ~/.bashrc'):${NC}"
        echo "   ddos-up         # Iniciar"
        echo "   ddos-down       # Parar"
        echo "   ddos-status     # Estado"
        echo "   ddos-logs       # Registos"
        echo "   ddos-check      # Verificar sistema"
        echo "   ddos-rebuild    # Reconstruir"
        echo ""
        echo -e "${GREEN}Machine Learning (Recomendado):${NC}"
        echo "   ddos-train-clean     # Treino rápido com datasets otimizados"
        echo "   ddos-validate        # Validar qualidade dos modelos"
        echo ""
        echo -e "${YELLOW}Outros comandos ML:${NC}"
        echo "   ddos-train      # Treinar modelo"
        echo "   ddos-train-simple    # Treino rápido"
        echo "   ddos-train-advanced  # Treino completo (pode dar erro de memória)"
        echo "   ddos-create-clean    # Recriar datasets otimizados"
        echo "   ddos-optimize   # Otimizar hiperparâmetros"
        echo "   ddos-benchmark  # Avaliar desempenho"
        echo "   ddos-analyze    # Analisar resultados"
        echo ""
    fi
    echo -e "${GREEN}${BOLD}Próximo passo recomendado:${NC}"
    echo "   1. source ~/.bashrc         # Ativar atalhos"
    echo "   2. ddos-train-clean         # Treinar modelos (rápido!)"
    echo ""
    echo -e "${BLUE}Documentação principal: README.md | docs/installation.md${NC}"
    echo -e "${BLUE}Para detalhes sobre datasets: setup/dataset-preparation/DATASET_SOURCES.md${NC}"
    echo -e "${BLUE}Registo desta execução: $LOG_FILE${NC}"
}

main() {
    # Inicializar log
    echo "DDoS Lab - Primeira execução iniciada em $(date)" > "$LOG_FILE"
    
    print_banner
    
    # Confirmar execução
    read -p "Executar primeira configuração do DDoS Lab? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Execução cancelada."
        exit 0
    fi
    
    # Verificar sistema
    if ! check_system; then
        echo ""
        read -p "O sistema precisa de configuração. Deseja executar o setup automático agora? [Y/n]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            run_setup
            echo ""
            log_warning "IMPORTANTE: Se foi adicionada a sua conta ao grupo 'docker', tem de fazer logout e login antes de continuar."
            log_info "Por favor, termine a sessão e volte a entrar no sistema para garantir que as permissões Docker estão ativas."
            read -p "Pressione Enter após fazer logout/login para continuar..."
        else
            log_error "Setup cancelado. O sistema pode não funcionar corretamente."
            exit 1
        fi
    fi
    
    # Garantir permissões de execução para todos os start.sh relevantes
    for service in ml-processor bgp-controller data-ingestion; do
        sh_path="$PROJECT_ROOT/src/$service/start.sh"
        if [ -f "$sh_path" ]; then
            chmod +x "$sh_path"
            log_info "Permissão de execução garantida para: $sh_path"
        fi
    done

    # Executar passos
    create_aliases
    check_datasets
    start_system
    show_instructions
}

# Trap para limpeza
trap 'log_error "Execução interrompida"; exit 1' INT TERM

# Executar apenas se chamado diretamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
