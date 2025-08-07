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
    echo -e "${GREEN}[OK]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "\n${CYAN}${BOLD}=== $1 ===${NC}" | tee -a "$LOG_FILE"
}

print_banner() {
    clear
    echo -e "${BLUE}${BOLD}"
    cat << 'EOF'
================================================================
    DDoS Detection & Mitigation Lab - First Run Setup
================================================================
EOF
    echo -e "${NC}"
    echo "Este script irá:"
    echo "• Verificar se o sistema está configurado"
    echo "• Executar setup se necessário"
    echo "• Criar aliases convenientes"
    echo "• Iniciar o sistema"
    echo "• Mostrar como usar"
    echo "" 
}

check_system() {
    log_step "Verificando Sistema"
    
    local need_setup=false
    
    # Docker
    if ! command -v docker &> /dev/null; then
        log_warning "Docker não encontrado"
        need_setup=true
    elif ! docker info &> /dev/null 2>&1; then
        log_warning "Docker não está funcionando"
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
        log_warning "Sistema precisa de configuração"
        return 1
    else
        log_success "Sistema já configurado"
        return 0
    fi
}

run_setup() {
    log_step "Executando Setup Completo"
    
    if [ -f "$PROJECT_ROOT/setup/install.sh" ]; then
        log_info "Executando setup simplificado..."
        chmod +x "$PROJECT_ROOT/setup/install.sh"
        "$PROJECT_ROOT/setup/install.sh"
    else
        log_error "Script de setup não encontrado: $PROJECT_ROOT/setup/install.sh"
        return 1
    fi
}

create_aliases() {
    log_step "Criando Aliases Convenientes"
    
    log_info "Criando aliases diretamente..."
    
    # Criar aliases diretamente no ~/.bashrc
    local alias_file="$HOME/.ddos_aliases"
    cat > "$alias_file" << 'EOF'
# DDoS Lab Aliases
alias ddos-up='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh up'
alias ddos-down='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh down'
alias ddos-status='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh status'
alias ddos-logs='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh logs'
alias ddos-check='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh test'
alias ddos-rebuild='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh rebuild'

# Machine Learning Aliases
alias ddos-train='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh train'
alias ddos-train-simple='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh train-simple'
alias ddos-train-advanced='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh train-advanced'
alias ddos-train-clean='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh train-clean'
alias ddos-create-clean='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh create-clean'
alias ddos-create-realistic='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh create-realistic'
alias ddos-train-realistic='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh train-realistic'
alias ddos-validate='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh validate'
alias ddos-optimize='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh optimize'
alias ddos-benchmark='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh benchmark'
alias ddos-analyze='cd ~/ddos-mitigation-lab && ./deployment/scripts/make.sh analyze'

# Demo e utilitários
alias ddos-demo='cd ~/ddos-mitigation-lab && python deployment/scripts/demo_modelo_ddos.py'
alias ddos-cd='cd ~/ddos-mitigation-lab'
EOF
    
    # Adicionar source ao bashrc se não existir
    if ! grep -q ".ddos_aliases" "$HOME/.bashrc" 2>/dev/null; then
        echo "source ~/.ddos_aliases" >> "$HOME/.bashrc"
    fi
    
    log_success "Aliases criados em $alias_file"
}

check_datasets() {
    log_step "Verificando e Preparando Datasets"
    
    local datasets_dir="$PROJECT_ROOT/src/datasets/integrated"
    local clean_datasets_dir="$PROJECT_ROOT/src/datasets/clean"
    
    # Verificar se datasets reais existem
    if [ ! -f "$datasets_dir/X_integrated_real.npy" ] || [ ! -f "$datasets_dir/y_integrated_real.npy" ]; then
        log_info "Datasets reais não encontrados. Iniciando download..."
        
        # Verificar se script de datasets existe
        if [ -f "$PROJECT_ROOT/setup/datasets.sh" ]; then
            log_info "Executando download automático de datasets..."
            
            # Tornar executável e executar
            chmod +x "$PROJECT_ROOT/setup/datasets.sh"
            if "$PROJECT_ROOT/setup/datasets.sh"; then
                log_success "Download de datasets concluído"
            else
                log_warning "Download padrão falhou. Tentando script robusto..."
                
                # Tentar script robusto como fallback
                if [ -f "$PROJECT_ROOT/setup/datasets_robust.sh" ]; then
                    chmod +x "$PROJECT_ROOT/setup/datasets_robust.sh"
                    if "$PROJECT_ROOT/setup/datasets_robust.sh"; then
                        log_success "Download robusto de datasets concluído"
                    else
                        log_warning "Download robusto falhou. Gerando dados de fallback..."
                        generate_fallback_data
                    fi
                else
                    log_warning "Script robusto não encontrado. Gerando dados de fallback..."
                    generate_fallback_data
                fi
            fi
        else
            log_warning "Script de download não encontrado. Gerando dados de fallback..."
            generate_fallback_data
        fi
    else
        log_success "Datasets reais já existem"
        
        # Mostrar estatísticas
        python3 -c "
import numpy as np
try:
    X = np.load('$datasets_dir/X_integrated_real.npy')
    y = np.load('$datasets_dir/y_integrated_real.npy')
    print(f'📊 Disponível: {X.shape[0]:,} amostras, {X.shape[1]} features')
    print(f'🎯 Ataques: {y.sum():,} ({y.mean():.1%})')
except Exception as e:
    print(f'⚠️ Erro ao ler datasets: {e}')
" 2>/dev/null || log_info "Datasets encontrados (validação falhou)"
    fi
    
    # Criar datasets limpos otimizados
    if [ ! -d "$clean_datasets_dir" ] || [ ! -f "$clean_datasets_dir/summary.json" ]; then
        log_info "Criando datasets limpos otimizados..."
        
        # Verificar se script existe
        if [ -f "$PROJECT_ROOT/deployment/scripts/create_clean_datasets.py" ]; then
            cd "$PROJECT_ROOT"
            python3 "$PROJECT_ROOT/deployment/scripts/create_clean_datasets.py" || {
                log_warning "Falha na criação dos datasets limpos, mas continuando..."
            }
            
            if [ -f "$clean_datasets_dir/summary.json" ]; then
                log_success "Datasets limpos otimizados criados"
                log_info "Disponíveis: CIC-DDoS2019 limpo, UNSW-NB15 limpo, Dataset integrado"
            else
                log_warning "Datasets limpos não foram criados corretamente"
            fi
        else
            log_warning "Script de criação de datasets limpos não encontrado"
        fi
    else
        log_success "Datasets limpos já existem"
    fi
}

generate_fallback_data() {
    log_info "Gerando dados de fallback para teste..."
    mkdir -p "$datasets_dir"
    
    python3 -c "
import numpy as np
import json
import os

print('Gerando dados sintéticos para teste...')
np.random.seed(42)
X = np.random.randn(5000, 20)
y = (X[:, 0] + X[:, 1] + np.random.randn(5000) * 0.1) > 0

datasets_dir = '$datasets_dir'
np.save(f'{datasets_dir}/X_integrated_real.npy', X)
np.save(f'{datasets_dir}/y_integrated_real.npy', y.astype(int))

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

print(f'✅ Dados de fallback gerados: {X.shape} amostras, {y.sum()} positivas')
print('⚠️ IMPORTANTE: Estes são dados sintéticos para teste')
"
    log_success "Dados de fallback gerados"
    log_warning "IMPORTANTE: Substitua por datasets reais executando: ./setup/datasets.sh"
}

start_system() {
    log_step "Iniciando Sistema DDoS"
    
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
    log_step "Sistema Pronto!"
    
    echo ""
    echo -e "${GREEN}${BOLD}✅ DDoS Mitigation Lab está rodando!${NC}"
    echo ""
    echo -e "${CYAN}📊 Dashboards disponíveis:${NC}"
    echo "   Grafana:    http://localhost:3000 (admin/admin123)"
    echo "   Prometheus: http://localhost:9090"
    echo ""
    echo -e "${CYAN}🔧 Comandos úteis:${NC}"
    echo "   ./deployment/scripts/make.sh status    # Ver status"
    echo "   ./deployment/scripts/make.sh logs      # Ver logs"
    echo "   ./deployment/scripts/make.sh down      # Parar sistema"
    echo "   ./deployment/scripts/make.sh help      # Ajuda completa"
    echo ""
    
    if [ -f "$HOME/.ddos_aliases" ]; then
        echo -e "${CYAN}Aliases disponíveis (após 'source ~/.bashrc'):${NC}"
        echo "   ddos-up         # Iniciar"
        echo "   ddos-down       # Parar"
        echo "   ddos-status     # Status"
        echo "   ddos-logs       # Logs"
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
        echo "   ddos-train-advanced  # Treino completo (pode dar erro memória)"
        echo "   ddos-create-clean    # Recriar datasets otimizados"
        echo "   ddos-optimize   # Otimizar hiperparâmetros"
        echo "   ddos-benchmark  # Avaliar performance"
        echo "   ddos-analyze    # Analisar resultados"
        echo ""
    fi
    
    echo -e "${GREEN}${BOLD}🚀 Próximo passo recomendado:${NC}"
    echo "   1. source ~/.bashrc         # Ativar aliases"
    echo "   2. ddos-train-clean         # Treinar modelos (rápido!)"
    echo ""
    echo -e "${BLUE}Documentação: README.md | docs/installation.md${NC}"
    echo -e "${BLUE}Log desta execução: $LOG_FILE${NC}"
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
        read -p "Sistema precisa de configuração. Executar setup agora? [Y/n]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            run_setup
            
            echo ""
            log_warning "IMPORTANTE: Faça logout e login antes de continuar"
            read -p "Pressione Enter após fazer logout/login..."
        else
            log_error "Setup cancelado. Sistema pode não funcionar corretamente."
            exit 1
        fi
    fi
    
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
