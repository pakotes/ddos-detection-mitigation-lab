#!/bin/bash
#
# DDoS Detection & Mitigation Lab- Setup Simplificado
# Script único para instalação completa
# 
# Uso: ./setup/install.sh
# Ou: curl -fsSL https://raw.githubusercontent.com/pakotes/ddos-detection-mitigation-lab/master/setup/install.sh | bash
#

set -euo pipefail

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'


# Variáveis
# Determinar a raiz do projeto (diretório pai de onde está este script)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_FILE="/tmp/ddos-setup.log"

log_info() { echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1" | tee -a "$LOG_FILE"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"; }

show_header() {
    echo -e "${CYAN}"
    echo "╔═════════════════════════════════════════════════════════════════════╗"
    echo "║            Ddos Detection & Mitigation - Setup Automático           ║"
    echo "║                          Linux Installation                         ║"
    echo "╚═════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_requirements() {
    log_info "Verificando requisitos do sistema..."
    
    # Verificar SO
    if ! grep -qi "ubuntu\|debian\|centos\|rocky\|rhel" /etc/os-release 2>/dev/null; then
        log_warning "SO não testado. Continuando..."
    fi
    
    # Verificar espaço em disco (mínimo 10GB)
    local available_gb=$(df / | awk 'NR==2 {print int($4/1024/1024)}')
    if [ "$available_gb" -lt 10 ]; then
        log_error "Espaço insuficiente. Necessário: 10GB, Disponível: ${available_gb}GB"
        exit 1
    fi
    
    # Verificar memória (mínimo 4GB)
    local memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$memory_gb" -lt 4 ]; then
        log_warning "Memória baixa (${memory_gb}GB). Recomendado: 8GB+"
    fi
    
    log_success "Requisitos verificados"
}

install_docker() {
    if command -v docker >/dev/null 2>&1; then
        log_info "Docker já instalado"
        return
    fi
    
    log_info "Instalando Docker..."
    
    # Detectar distribuição
    if command -v apt >/dev/null 2>&1; then
        # Ubuntu/Debian
        sudo apt update
        sudo apt install -y curl ca-certificates gnupg lsb-release
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt update
        sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    elif command -v yum >/dev/null 2>&1; then
        # CentOS/Rocky/RHEL
        sudo yum install -y yum-utils
        sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
        sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        sudo systemctl start docker
        sudo systemctl enable docker
    else
        log_error "Distribuição não suportada para instalação automática do Docker"
        exit 1
    fi
    
    # Configurar usuário
    sudo usermod -aG docker $USER
    log_success "Docker instalado. IMPORTANTE: Faça logout/login para aplicar permissões"
}

install_python() {
    if python3 --version >/dev/null 2>&1; then
        log_info "Python já instalado: $(python3 --version)"
        return
    fi
    
    log_info "Instalando Python..."
    
    if command -v apt >/dev/null 2>&1; then
        sudo apt update
        sudo apt install -y python3 python3-pip python3-venv
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y python3 python3-pip
    fi
    
    log_success "Python instalado"
}

install_dependencies() {
    log_info "Instalando dependências ML..."
    
    # Criar requirements temporário se não existir
    log_info "Atualizando pip, setuptools e wheel..."
    python3 -m pip install --upgrade pip setuptools wheel --user

    if [ ! -f "${PROJECT_ROOT}/requirements.txt" ]; then
        cat > /tmp/requirements.txt << 'EOF'
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.3.0
seaborn>=0.11.0
jupyter>=1.0.0
plotly>=5.0.0
EOF
        pip3 install -r /tmp/requirements.txt --user
    else
        pip3 install -r "${PROJECT_ROOT}/requirements.txt" --user
    fi
    
    log_success "Dependências ML instaladas"
}

setup_project() {
    log_info "Configurando projeto..."
    
    # Criar estrutura de diretórios
    mkdir -p "${PROJECT_ROOT}/src/datasets/"{raw,integrated}
    mkdir -p "${PROJECT_ROOT}/src/models"
    mkdir -p "${PROJECT_ROOT}/logs"
    
    # Dar permissões aos scripts
    find "${PROJECT_ROOT}" -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    
    log_success "Projeto configurado"
}

main() {
    show_header
    
    log_info "Iniciando setup automático..."
    
    check_requirements
    install_docker
    install_python
    install_dependencies
    setup_project
    
    log_success "Setup concluído com sucesso!"
    echo ""
    echo -e "${CYAN}Logs salvos em: $LOG_FILE${NC}"
}

# Executar se chamado diretamente
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
