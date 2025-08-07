#!/bin/bash

# Script de deploy distribuído para sistema de mitigação DDoS
# Usage: ./deploy-distributed.sh [machine1|machine2|machine3|all]

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configurações das máquinas
MACHINE1_IP=${MACHINE1_IP:-"192.168.1.10"}
MACHINE2_IP=${MACHINE2_IP:-"192.168.1.11"}
MACHINE3_IP=${MACHINE3_IP:-"192.168.1.12"}

# Função de log
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Verificar se Docker está instalado
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker não está instalado. Por favor, instale o Docker primeiro."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose não está instalado. Por favor, instale o Docker Compose primeiro."
        exit 1
    fi
}

# Criar rede Docker compartilhada
create_network() {
    log "Criando rede Docker compartilhada..."
    
    # Verificar se a rede já existe
    if docker network ls | grep -q "ddos-mitigation-network"; then
        warn "Rede ddos-mitigation-network já existe, removendo..."
        docker network rm ddos-mitigation-network || true
        sleep 2
    fi
    
    # Criar nova rede
    docker network create \
        --driver bridge \
        --subnet=172.20.0.0/16 \
        --ip-range=172.20.240.0/20 \
        ddos-mitigation-network
    
    log "Rede ddos-mitigation-network criada com sucesso"
}

# Deploy Machine 1 - Kafka e Redis
deploy_machine1() {
    log "Iniciando deploy da Machine 1 (Kafka + ZooKeeper + Redis)..."
    
    # Exportar variáveis de ambiente
    export MACHINE1_IP
    export MACHINE2_IP
    export MACHINE3_IP
    
    # Parar containers existentes
    docker-compose -f docker-compose.machine1.yml down || true
    
    # Remover volumes antigos se necessário
    if [[ "$1" == "--clean" ]]; then
        warn "Removendo volumes antigos da Machine 1..."
        docker volume rm distributed_kafka-data distributed_zookeeper-data distributed_redis-data || true
    fi
    
    # Iniciar serviços
    docker-compose -f docker-compose.machine1.yml up -d
    
    # Aguardar serviços ficarem prontos
    log "Aguardando Kafka e ZooKeeper iniciarem..."
    sleep 30
    
    # Verificar status
    if docker-compose -f docker-compose.machine1.yml ps | grep -q "Up"; then
        log "Machine 1 deployada com sucesso!"
        
        # Criar tópicos Kafka
        log "Criando tópicos Kafka..."
        docker-compose -f docker-compose.machine1.yml exec kafka kafka-topics.sh \
            --create --topic traffic-logs --partitions 6 --replication-factor 1 \
            --bootstrap-server localhost:9092 || warn "Tópico traffic-logs pode já existir"
            
        docker-compose -f docker-compose.machine1.yml exec kafka kafka-topics.sh \
            --create --topic malicious-ips --partitions 3 --replication-factor 1 \
            --bootstrap-server localhost:9092 || warn "Tópico malicious-ips pode já existir"
            
        docker-compose -f docker-compose.machine1.yml exec kafka kafka-topics.sh \
            --create --topic bgp-updates --partitions 3 --replication-factor 1 \
            --bootstrap-server localhost:9092 || warn "Tópico bgp-updates pode já existir"
            
        log "Tópicos Kafka criados com sucesso"
    else
        error "Falha no deploy da Machine 1"
        exit 1
    fi
}

# Deploy Machine 2 - ML Processor e Data Ingestion
deploy_machine2() {
    log "Iniciando deploy da Machine 2 (ML Processor + Data Ingestion)..."
    
    # Exportar variáveis de ambiente
    export MACHINE1_IP
    export MACHINE2_IP
    export MACHINE3_IP
    
    # Parar containers existentes
    docker-compose -f docker-compose.machine2.yml down || true
    
    # Remover volumes antigos se necessário
    if [[ "$1" == "--clean" ]]; then
        warn "Removendo volumes antigos da Machine 2..."
        docker volume rm distributed_ml-models distributed_data-ingestion-logs distributed_ml-processor-logs || true
    fi
    
    # Aguardar Kafka estar disponível
    log "Verificando conectividade com Kafka..."
    timeout 60 bash -c "until nc -z ${MACHINE1_IP} 9092; do sleep 1; done" || {
        error "Não foi possível conectar ao Kafka na Machine 1"
        exit 1
    }
    
    # Iniciar serviços
    docker-compose -f docker-compose.machine2.yml up -d
    
    # Aguardar serviços ficarem prontos
    log "Aguardando serviços da Machine 2 iniciarem..."
    sleep 20
    
    # Verificar status
    if docker-compose -f docker-compose.machine2.yml ps | grep -q "Up"; then
        log "Machine 2 deployada com sucesso!"
    else
        error "Falha no deploy da Machine 2"
        exit 1
    fi
}

# Deploy Machine 3 - BGP Controller e Monitoring
deploy_machine3() {
    log "Iniciando deploy da Machine 3 (BGP Controller + Monitoring)..."
    
    # Exportar variáveis de ambiente
    export MACHINE1_IP
    export MACHINE2_IP
    export MACHINE3_IP
    
    # Parar containers existentes
    docker-compose -f docker-compose.machine3.yml down || true
    
    # Remover volumes antigos se necessário
    if [[ "$1" == "--clean" ]]; then
        warn "Removendo volumes antigos da Machine 3..."
        docker volume rm distributed_bgp-logs distributed_prometheus-data distributed_grafana-data || true
    fi
    
    # Aguardar Kafka estar disponível
    log "Verificando conectividade com Kafka..."
    timeout 60 bash -c "until nc -z ${MACHINE1_IP} 9092; do sleep 1; done" || {
        error "Não foi possível conectar ao Kafka na Machine 1"
        exit 1
    }
    
    # Iniciar serviços
    docker-compose -f docker-compose.machine3.yml up -d
    
    # Aguardar serviços ficarem prontos
    log "Aguardando serviços da Machine 3 iniciarem..."
    sleep 25
    
    # Verificar status
    if docker-compose -f docker-compose.machine3.yml ps | grep -q "Up"; then
        log "Machine 3 deployada com sucesso!"
        
        # Mostrar URLs dos serviços
        info "Serviços disponíveis:"
        info "  - Grafana: http://${MACHINE3_IP}:3000 (admin/admin123)"
        info "  - Prometheus: http://${MACHINE3_IP}:9090"
        info "  - AlertManager: http://${MACHINE3_IP}:9093"
    else
        error "Falha no deploy da Machine 3"
        exit 1
    fi
}

# Verificar status de todas as máquinas
check_status() {
    log "Verificando status do sistema distribuído..."
    
    info "=== Machine 1 Status ==="
    docker-compose -f docker-compose.machine1.yml ps
    
    info "=== Machine 2 Status ==="
    docker-compose -f docker-compose.machine2.yml ps
    
    info "=== Machine 3 Status ==="
    docker-compose -f docker-compose.machine3.yml ps
    
    info "=== Rede Docker ==="
    docker network ls | grep ddos-mitigation-network
    
    info "=== Conectividade Kafka ==="
    if nc -z ${MACHINE1_IP} 9092; then
        log "Kafka acessível em ${MACHINE1_IP}:9092"
    else
        error "Kafka não acessível em ${MACHINE1_IP}:9092"
    fi
    
    if nc -z ${MACHINE1_IP} 6379; then
        log "Redis acessível em ${MACHINE1_IP}:6379"
    else
        error "Redis não acessível em ${MACHINE1_IP}:6379"
    fi
}

# Parar todos os serviços
stop_all() {
    log "Parando todos os serviços distribuídos..."
    
    docker-compose -f docker-compose.machine3.yml down
    docker-compose -f docker-compose.machine2.yml down  
    docker-compose -f docker-compose.machine1.yml down
    
    log "Todos os serviços foram parados"
}

# Limpeza completa
clean_all() {
    log "Realizando limpeza completa do sistema distribuído..."
    
    # Parar todos os serviços
    stop_all
    
    # Remover volumes
    warn "Removendo todos os volumes..."
    docker volume rm distributed_kafka-data distributed_zookeeper-data distributed_redis-data || true
    docker volume rm distributed_ml-models distributed_data-ingestion-logs distributed_ml-processor-logs || true
    docker volume rm distributed_bgp-logs distributed_prometheus-data distributed_grafana-data || true
    
    # Remover rede
    docker network rm ddos-mitigation-network || true
    
    log "Limpeza completa finalizada"
}

# Menu principal
show_help() {
    echo -e "${BLUE}Sistema de Mitigação DDoS - Deploy Distribuído${NC}"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  machine1      Deploy Machine 1 (Kafka + ZooKeeper + Redis)"
    echo "  machine2      Deploy Machine 2 (ML Processor + Data Ingestion)"
    echo "  machine3      Deploy Machine 3 (BGP Controller + Monitoring)"
    echo "  all           Deploy todas as máquinas sequencialmente"
    echo "  status        Verificar status de todos os serviços"
    echo "  stop          Parar todos os serviços"
    echo "  clean         Limpeza completa (remove volumes e dados)"
    echo "  help          Mostrar esta ajuda"
    echo ""
    echo "Options:"
    echo "  --clean       Remove volumes antigos antes do deploy"
    echo ""
    echo "Variáveis de ambiente:"
    echo "  MACHINE1_IP   IP da Machine 1 (default: 192.168.1.10)"
    echo "  MACHINE2_IP   IP da Machine 2 (default: 192.168.1.11)"
    echo "  MACHINE3_IP   IP da Machine 3 (default: 192.168.1.12)"
    echo ""
    echo "Exemplos:"
    echo "  $0 all                    # Deploy completo"
    echo "  $0 machine1 --clean      # Deploy Machine 1 com limpeza"
    echo "  MACHINE1_IP=10.0.0.1 $0 machine2  # Deploy com IP customizado"
}

# Script principal
main() {
    check_docker
    
    case "${1:-help}" in
        "machine1")
            create_network
            deploy_machine1 "$2"
            ;;
        "machine2")
            deploy_machine2 "$2"
            ;;
        "machine3")
            deploy_machine3 "$2"
            ;;
        "all")
            create_network
            deploy_machine1 "$2"
            deploy_machine2 "$2"
            deploy_machine3 "$2"
            check_status
            ;;
        "status")
            check_status
            ;;
        "stop")
            stop_all
            ;;
        "clean")
            clean_all
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Executar função principal
main "$@"
