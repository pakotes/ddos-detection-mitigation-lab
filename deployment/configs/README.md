# Deployment Distribuído - 3 Máquinas

Este diretório contém as configurações para deployment do sistema em 3 máquinas separadas para ambiente de produção.

## Arquitetura

- **Máquina 1**: Infraestrutura (Kafka, Zookeeper, Redis)
- **Máquina 2**: Processamento (ML Processor, Data Ingestion)  
- **Máquina 3**: Controle e Monitoramento (BGP Controller, Prometheus, Grafana)

## Configuração

### 1. Variáveis de Ambiente
```bash
export MACHINE1_IP=192.168.1.10
export MACHINE2_IP=192.168.1.11
export MACHINE3_IP=192.168.1.12
```

### 2. Pré-requisitos
- Docker e Docker Compose instalados em todas as máquinas
- Conectividade de rede entre as máquinas
- Portas abertas: 9092, 2181, 6379, 8000, 8001, 8002, 3000, 9090, 9093

## Deploy

### Automático (Recomendado)
```bash
# Deploy completo
./deploy-distributed.sh all

# Ou individual
./deploy-distributed.sh machine1  # Primeiro
./deploy-distributed.sh machine2  # Segundo  
./deploy-distributed.sh machine3  # Terceiro
```

### Via Makefile
```bash
make deploy-distributed        # Deploy completo
make deploy-machine1           # Individual
make status-distributed        # Verificar status
make stop-distributed          # Parar sistema
```

## Acesso aos Serviços

- **Grafana**: http://${MACHINE3_IP}:3000 (admin/admin123)
- **Prometheus**: http://${MACHINE3_IP}:9090
- **Kafka**: ${MACHINE1_IP}:9092
- **Redis**: ${MACHINE1_IP}:6379

## Desenvolvimento vs Produção

- **Desenvolvimento**: Use `docker-compose.yml` na raiz (1 máquina)
- **Produção**: Use este deployment distribuído (3 máquinas)
