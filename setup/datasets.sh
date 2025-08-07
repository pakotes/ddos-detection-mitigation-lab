#!/bin/bash
#
# DDoS Detection Lab - Download de Datasets Académicos
# Script para facilitar o download de CIC-DDoS2019 e NF-UNSW-NB15-v3
# 
# IMPORTANTE: Os datasets requerem download manual devido a políticas académicas
#
# Uso: ./setup/datasets.sh [--check-only] [--help]
#

set -euo pipefail

# Função para mostrar estatísticas de download
show_download_summary() {
    local dataset_name="$1"
    local file_path="$2"
    
    if [ -f "$file_path" ]; then
        local file_size=$(du -h "$file_path" | cut -f1)
        local file_name=$(basename "$file_path")
        
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║                    DOWNLOAD CONCLUÍDO                     ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo -e "${GREEN}Dataset:${NC} $dataset_name"
        echo -e "${GREEN}Arquivo:${NC} $file_name"
        echo -e "${GREEN}Tamanho:${NC} $file_size"
        echo -e "${GREEN}Local:${NC} $file_path"
        echo ""
        return 0
    else
        echo -e "${RED}✗ Arquivo não encontrado: $file_path${NC}"
        return 1
    fi
}

# Verificar dependências de download
check_download_tools() {
    echo -e "${BLUE}Verificando ferramentas de download...${NC}"
    
    if ! command -v wget >/dev/null 2>&1; then
        log_error "wget não está instalado!"
        log_info "No Ubuntu/Debian: sudo apt-get install wget"
        log_info "No CentOS/RHEL: sudo yum install wget"
        return 1
    fi
    
    if ! command -v curl >/dev/null 2>&1; then
        log_warning "curl não está instalado (recomendado como backup)"
    fi
    
    echo -e "${GREEN}✓ Ferramentas de download disponíveis${NC}"
    return 0
}
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Variáveis
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASETS_DIR="${PROJECT_ROOT}/src/datasets"
TEMP_DIR="/tmp/ddos-datasets"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_header() {
    clear
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════════════╗"
    echo "║                          DDoS DETECTION LAB                              ║"
    echo "║                        Download de Datasets                              ║"
    echo "║                                                                           ║"
    echo "║    📊 CIC-DDoS2019: Comprehensive DDoS Attack Dataset                    ║"
    echo "║    🌐 NF-UNSW-NB15-v3: NetFlow Network Intrusion Dataset                ║"
    echo "║                                                                           ║"
    echo "║    Status: Downloading with visual progress indicators                   ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

check_existing_datasets() {
    log_info "Verificando estrutura dos datasets..."
    
    local dataset_prep_dir="${SCRIPT_DIR}/dataset-preparation"
    
    echo ""
    echo -e "${CYAN}=== ESTADO ATUAL DOS DATASETS ===${NC}"
    
    # Verificar CIC-DDoS2019
    local cic_dir="${dataset_prep_dir}/CIC-DDoS2019"
    if [ -d "$cic_dir" ]; then
        local dir_0112="${cic_dir}/01-12"
        local dir_0311="${cic_dir}/03-11"
        local csv_count=0
        
        if [ -d "$dir_0112" ]; then
            csv_count=$((csv_count + $(find "$dir_0112" -name "*.csv" | wc -l)))
        fi
        
        if [ -d "$dir_0311" ]; then
            csv_count=$((csv_count + $(find "$dir_0311" -name "*.csv" | wc -l)))
        fi
        
        if [ $csv_count -eq 13 ]; then
            echo -e "${GREEN}✓ CIC-DDoS2019: Completo (13 ficheiros CSV)${NC}"
            if [ -d "$dir_0112" ]; then
                local files_0112=$(find "$dir_0112" -name "*.csv" | wc -l)
                echo -e "  📁 01-12/: $files_0112 ficheiros"
            fi
            if [ -d "$dir_0311" ]; then
                local files_0311=$(find "$dir_0311" -name "*.csv" | wc -l)
                echo -e "  📁 03-11/: $files_0311 ficheiros"
            fi
        else
            echo -e "${YELLOW}⚠ CIC-DDoS2019: Incompleto ($csv_count/13 ficheiros)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ CIC-DDoS2019: Ausente${NC}"
    fi
    
    # Verificar NF-UNSW-NB15-v3
    local nf_dir="${dataset_prep_dir}/NF-UNSW-NB15-v3"
    if [ -d "$nf_dir" ]; then
        local csv_count=$(find "$nf_dir" -maxdepth 1 -name "*.csv" | wc -l)
        
        if [ $csv_count -eq 2 ]; then
            echo -e "${GREEN}✓ NF-UNSW-NB15-v3: Completo (2 ficheiros CSV)${NC}"
            
            if [ -f "${nf_dir}/NetFlow_v3_Features.csv" ]; then
                local size=$(du -h "${nf_dir}/NetFlow_v3_Features.csv" | cut -f1)
                echo -e "  📄 NetFlow_v3_Features.csv: $size"
            fi
            
            if [ -f "${nf_dir}/NF-UNSW-NB15-v3.csv" ]; then
                local size=$(du -h "${nf_dir}/NF-UNSW-NB15-v3.csv" | cut -f1)
                echo -e "  📄 NF-UNSW-NB15-v3.csv: $size"
            fi
        else
            echo -e "${YELLOW}⚠ NF-UNSW-NB15-v3: Incompleto ($csv_count/2 ficheiros)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ NF-UNSW-NB15-v3: Ausente${NC}"
    fi
    
    echo ""
print(f'🎯 Ataques: {y.sum():,} ({y.mean():.1%})')
" 2>/dev/null || log_info "Datasets encontrados (validação Python falhou)"
        
        echo ""
        read -p "Deseja fazer re-download? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Usando datasets existentes"
            return 0
        fi
    fi
    
    return 1
}

download_cicddos2019() {
    log_info "CIC-DDoS2019 - Dataset especializado em ataques DDoS"
    
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                   CIC-DDoS2019 DATASET                     ║${NC}"
    echo -e "${BLUE}║              Estrutura: 13 ficheiros CSV                   ║${NC}"
    echo -e "${BLUE}║          Directórios: 01-12/ e 03-11/                      ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    
    log_warning "IMPORTANTE: Download manual obrigatório devido a políticas académicas"
    echo ""
    
    echo -e "${YELLOW}INSTRUÇÕES DE DOWNLOAD MANUAL:${NC}"
    echo ""
    echo -e "${GREEN}Fonte Oficial:${NC}"
    echo "  1. Aceder: https://www.unb.ca/cic/datasets/ddos-2019.html"
    echo "  2. Preencher formulário de registo académico"
    echo "  3. Descarregar: DDoS2019.zip (~2.3 GB)"
    echo "  4. Extrair para: setup/dataset-preparation/CIC-DDoS2019/"
    echo ""
    echo -e "${GREEN}Alternativa Kaggle:${NC}"
    echo "  • https://www.kaggle.com/datasets/aymenabb/ddos-evaluation-dataset-cic-ddos2019"
    echo ""
    echo -e "${GREEN}Estrutura esperada após extração:${NC}"
    echo "  CIC-DDoS2019/"
    echo "  ├── 01-12/     (6 ficheiros CSV: DrDoS_SNMP, DrDoS_SSDP, DrDoS_UDP, Syn, TFTP, UDPLag)"
    echo "  └── 03-11/     (7 ficheiros CSV: LDAP, MSSQL, NetBIOS, Portmap, Syn, UDP, UDPLag)"
    echo ""
            
            if [[ "$url" == *".csv"* ]]; then
                # Download direto CSV
                if wget --progress=bar:force:noscroll \
                       --show-progress \
                       --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
                       --timeout=30 \
                       --tries=3 \
                       -O "${TEMP_DIR}/cicddos2019/DDoS2019.csv" \
                       "$url"; then
                    echo ""
                    log_success "CIC-DDoS2019 CSV baixado com sucesso!"
                    local file_size=$(du -h "${TEMP_DIR}/cicddos2019/DDoS2019.csv" | cut -f1)
                    log_info "Arquivo: DDoS2019.csv (${file_size})"
                    success=true
                    break
                fi
            else
                # Download ZIP
                local filename=$(basename "$url")
                if wget --progress=bar:force:noscroll \
                       --show-progress \
                       --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
                       --timeout=30 \
                       --tries=3 \
                       -O "${TEMP_DIR}/cicddos2019/$filename" \
                       "$url"; then
                    echo ""
                    log_success "Arquivo $filename baixado com sucesso!"
                    local file_size=$(du -h "${TEMP_DIR}/cicddos2019/$filename" | cut -f1)
                    log_info "Arquivo: $filename (${file_size})"
                    success=true
                    break
                fi
            fi
        else
            echo -e "${RED}✗ URL não disponível${NC}"
        fi
        
        echo ""
        log_warning "Tentativa ${attempt} falhou, tentando próxima fonte..."
        ((attempt++))
        echo ""
    done
    
    if [ "$success" = false ]; then
        log_warning "Download automático falhou"
        log_info "Para download manual:"
        echo "1. Visite: https://www.unb.ca/cic/datasets/ddos-2019.html"
        echo "2. Baixe os arquivos CSV"
        echo "3. Coloque em: ${DATASETS_DIR}/raw/cicddos2019/"
        return 1
    fi
    
    log_success "CIC-DDoS2019 baixado"
    return 0
}

download_unsw_nb15() {
    log_info "NF-UNSW-NB15-v3 - Dataset NetFlow para detecção de intrusões"
    
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                 NF-UNSW-NB15-v3 DATASET                    ║${NC}"
    echo -e "${BLUE}║             Estrutura: 2 ficheiros CSV                     ║${NC}"
    echo -e "${BLUE}║          NetFlow v3 Features + Dataset Principal           ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    
    log_warning "IMPORTANTE: Download manual recomendado para garantir integridade"
    echo ""
    
    echo -e "${YELLOW}INSTRUÇÕES DE DOWNLOAD MANUAL:${NC}"
    echo ""
    echo -e "${GREEN}Fonte Oficial (University of Queensland):${NC}"
    echo "  1. Aceder: https://espace.library.uq.edu.au/view/UQ:6e0eda1"
    echo "  2. Descarregar: NF-UNSW-NB15-v3.zip (~800 MB)"
    echo "  3. Extrair para: setup/dataset-preparation/NF-UNSW-NB15-v3/"
    echo ""
    echo -e "${GREEN}Alternativa Kaggle:${NC}"
    echo "  • https://www.kaggle.com/datasets/ndayisabae/nf-unsw-nb15-v3"
    echo ""
    echo -e "${GREEN}Estrutura esperada após extração:${NC}"
    echo "  NF-UNSW-NB15-v3/"
    echo "  ├── NetFlow_v3_Features.csv    (Ficheiro de características NetFlow)"
    echo "  └── NF-UNSW-NB15-v3.csv        (Dataset principal)"
    echo ""
    echo -e "${GREEN}Informações Técnicas:${NC}"
    echo "  • Autores: Luay, Majed; Layeghy, Siamak; et al."
    echo "  • DOI: https://doi.org/10.48610/6e0eda1"
    echo "  • 53 características NetFlow extraídas"
    echo "  • Etiquetas binárias e multi-classe"
    echo "  • Características temporais incluídas"
    echo ""
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║             FALLBACK - NF-UNSW-NB15-v3 Tradicional       ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
    
    log_warning "NetFlow version falhou, tentando versão tradicional..."
    local success=false
    for url in "${urls[@]}"; do
        if [[ "$url" == *"cloudstor"* ]]; then
            echo ""
            log_info "Tentando download NF-UNSW-NB15-v3 tradicional..."
            echo -e "${YELLOW}Iniciando download...${NC}"
            
            if wget --progress=bar:force:noscroll \
                   --show-progress \
                   --timeout=30 \
                   --tries=3 \
                   -O "${TEMP_DIR}/nf-unsw-nb15-v3/nf-unsw-nb15-v3.zip" \
                   "$url"; then
                echo ""
                log_success "Download tradicional concluído!"
                local file_size=$(du -h "${TEMP_DIR}/nf-unsw-nb15-v3/nf-unsw-nb15-v3.zip" | cut -f1)
                log_info "Arquivo: nf-unsw-nb15-v3.zip (${file_size})"
                success=true
                break
            fi
        fi
    done
    
    if [ "$success" = false ]; then
        log_warning "Todos os downloads NF-UNSW-NB15-v3 falharam"
        log_info "Para download manual:"
        echo "1. NetFlow v3 (recomendado):"
        echo "   URL: https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f/download"
        echo "   Descrição: NF-UNSW-NB15-v3 (NetFlow format)"
        echo "2. Tradicional: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
        echo "3. Coloque em: ${DATASETS_DIR}/raw/nf-unsw-nb15-v3/"
        echo ""
        echo "NOTA: O NF-UNSW-NB15-v3 é preferível por usar formato NetFlow"
        return 1
    fi
    
    return 0
}

process_datasets() {
    log_info "Processando datasets..."
    
    # Criar script Python para processamento
    cat > "${TEMP_DIR}/process.py" << 'EOF'
import numpy as np
import pandas as pd
import os
from pathlib import Path

def create_synthetic_data():
    """Criar dados sintéticos para desenvolvimento"""
    print("📊 Criando dados sintéticos...")
    
    np.random.seed(42)
    n_samples = 200000
    n_features = 50
    
    # Gerar dados base
    X = np.random.randn(n_samples, n_features)
    
    # Simular ataques DDoS (5.4% como no UNSW real)
    attack_ratio = 0.054
    n_attacks = int(n_samples * attack_ratio)
    
    # Padrões de ataque
    X[-n_attacks:, :5] = np.random.exponential(2, (n_attacks, 5))
    X[-n_attacks:, 5:10] = np.random.uniform(5, 15, (n_attacks, 5))
    
    # Labels
    y = np.zeros(n_samples)
    y[-n_attacks:] = 1
    
    return X, y

def main():
    datasets_dir = Path("src/datasets")
    integrated_dir = datasets_dir / "integrated"
    integrated_dir.mkdir(parents=True, exist_ok=True)
    
    # Por enquanto, criar dados sintéticos
    # TODO: Processar dados reais quando disponíveis
    X, y = create_synthetic_data()
    
    # Salvar
    np.save(integrated_dir / "X_integrated_real.npy", X)
    np.save(integrated_dir / "y_integrated_real.npy", y)
    
    # Criar metadata
    metadata = {
        "source": "Synthetic (Development)",
        "n_samples": len(y),
        "n_features": X.shape[1],
        "class_distribution": {
            "normal": int((y == 0).sum()),
            "attack": int((y == 1).sum())
        }
    }
    
    import json
    with open(integrated_dir / "metadata_real.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Features names
    feature_names = [f"feature_{i:02d}" for i in range(X.shape[1])]
    with open(integrated_dir / "feature_names_real.txt", "w") as f:
        f.write("\n".join(feature_names))
    
    print(f"✅ Dados processados: {X.shape[0]:,} amostras, {X.shape[1]} features")
    print(f"🎯 {int(y.sum()):,} ataques ({y.mean():.1%})")

if __name__ == "__main__":
    main()
EOF
    
    # Executar processamento
    cd "$PROJECT_ROOT"
    python3 "${TEMP_DIR}/process.py"
    
    log_success "Datasets processados"
}

cleanup() {
    log_info "Limpando arquivos temporários..."
    rm -rf "$TEMP_DIR" 2>/dev/null || true
    log_success "Limpeza concluída"
}

main() {
    log_info "Sistema de Preparação de Datasets para Investigação DDoS"
    log_info "Universidade do Porto - Mestrado em Segurança de Redes"
    echo ""
    
    check_existing_datasets
    
    echo -e "${CYAN}=== INSTRUÇÕES DE DOWNLOAD MANUAL ===${NC}"
    echo "Para obter os datasets completos, siga as instruções no ficheiro:"
    echo -e "${YELLOW}setup/dataset-preparation/DATASET_SOURCES.md${NC}"
    echo ""
    echo "Ou execute as funções específicas de download:"
    echo "  ./datasets.sh download_cicddos2019"
    echo "  ./datasets.sh download_unsw_nb15"
    echo "  ./datasets.sh check"
    echo ""
    
    # Verificar se é chamada de função específica
    if [ $# -gt 0 ]; then
        case "$1" in
            "download_cicddos2019")
                download_cicddos2019
                ;;
            "download_unsw_nb15")
                download_unsw_nb15
                ;;
            "check")
                # Função check já foi executada acima
                ;;
            *)
                log_error "Função desconhecida: $1"
                echo "Funções disponíveis: download_cicddos2019, download_unsw_nb15, check"
                exit 1
                ;;
        esac
    fi
}

# Trap para limpeza em caso de interrupção
trap cleanup EXIT

# Executar se chamado diretamente
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
