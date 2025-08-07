# DDoS Detection & Mitigation Lab

Sistema profissional de detecção e mitigação de ataques DDoS baseado em machine learning e arquitetura distribuída.

## Visão Geral

Este laboratório implementa uma solução completa para detecção e mitigação de ataques DDoS usando:

- **Machine Learning Híbrido**: XGBoost + Isolation Forest para detecção em tempo real
- **Datasets Profissionais**: CIC-DDoS2019 e NF-UNSW-NB15-v3 para treino e validação
- **Arquitetura Distribuída**: Containers Docker orquestrados com Kafka e Redis
- **Mitigação Automática**: Bloqueio BGP cooperativo e rate limiting adaptativo
- **Monitorização Completa**: Dashboards Grafana e métricas Prometheus

## Quick Start

### Instalação Completa Automática (Linux)
```bash
git clone https://github.com/pakotes/ddos-mitigation-lab.git
cd ddos-mitigation-lab
chmod +x first-run.sh
./first-run.sh
```

**O que o first-run.sh faz automaticamente:**
1. **Verificação do sistema** (Docker, Python, dependências)
2. **Setup inicial** se necessário (instalação e configuração)
3. **Download de datasets** (CIC-DDoS2019, NF-UNSW-NB15-v3)
4. **Criação de aliases** convenientes (ddos-*)
5. **Inicialização do sistema** completo

### Instalação Manual (Alternativa)
```bash
# 1. Setup do sistema
./setup/install.sh

# 2. Download de datasets 
./setup/datasets.sh

# 3. Inicialização
./deployment/scripts/make.sh up
```

### Acesso aos Dashboards
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090

## Arquitetura do Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Ingestion │    │   ML Processor  │    │ BGP Controller  │
│     :8002       │◄───┤      :8000      ├───►│     :8001       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  ▼
          ┌─────────────────┬─────────────────┬─────────────────┐
          │     Kafka       │     Redis       │   Monitoring   │
          │     :9092       │     :6379       │ Grafana :3000   │
          │                 │                 │ Prometheus :9090│
          └─────────────────┴─────────────────┴─────────────────┘
```

## Componentes Principais

### Machine Learning Pipeline
- **Modelos Híbridos**: Ensemble de XGBoost e Isolation Forest
- **77 Features**: Extração avançada de características de rede
- **Hierarquia Inteligente**: Carregamento automático de modelos otimizados
- **Treino Contínuo**: Adaptação a novos padrões de ataque

### Datasets Profissionais
- **CIC-DDoS2019**: Dataset oficial com 12 tipos de ataques DDoS
- **NF-UNSW-NB15-v3**: Dataset complementar para detecção de anomalias
- **Download Automático**: Scripts integrados para obtenção dos dados
- **Pré-processamento**: Feature engineering automatizado

### Sistema de Mitigação
- **BGP Blackholing**: Anúncios BGP para bloqueio cooperativo
- **Rate Limiting**: Controle adaptativo baseado em ML
- **Reputation System**: Sistema de reputação distribuído via Redis
- **Resposta Graduada**: Mitigação proporcional à severidade

## Comandos Principais

### Primeiro Setup (Execute UMA vez)
```bash
./first-run.sh                           # Setup completo automático
# OU manualmente:
./setup/install.sh                       # Instalar sistema  
./setup/datasets.sh                      # Baixar datasets
```

### Operação Diária
```bash
# Gerenciamento do sistema
./deployment/scripts/make.sh up          # Iniciar todos os serviços
./deployment/scripts/make.sh down        # Parar sistema
./deployment/scripts/make.sh status      # Status dos containers
./deployment/scripts/make.sh logs        # Logs em tempo real

# Machine Learning
./deployment/scripts/make.sh train       # Treinar modelos híbridos
./deployment/scripts/make.sh optimize    # Otimizar hiperparâmetros
./deployment/scripts/make.sh benchmark   # Avaliar performance

# Diagnóstico e manutenção
./deployment/scripts/make.sh clean       # Limpeza completa
./deployment/scripts/make.sh rebuild     # Reconstruir sistema
./deployment/scripts/make.sh help        # Ajuda completa
```

### Aliases Convenientes (Após first-run.sh)
```bash
ddos-up          # = ./deployment/scripts/make.sh up
ddos-down        # = ./deployment/scripts/make.sh down  
ddos-status      # = ./deployment/scripts/make.sh status
ddos-logs        # = ./deployment/scripts/make.sh logs
ddos-train       # = ./deployment/scripts/make.sh train
ddos-demo        # = python demo_modelo_ddos.py
```

### Demonstração do Modelo
```bash
# Ver o modelo em ação (após treino)
python deployment/scripts/demo_modelo_ddos.py
# OU usando alias:
ddos-demo
```

## Requisitos do Sistema

### Mínimos
- **Sistema Operacional**: Linux (Rocky Linux 10.0+ recomendado)
- **RAM**: 4GB (8GB recomendado para treino de modelos)
- **Disco**: 10GB livres
- **Rede**: Conexão com internet para download de datasets

### Recomendados
- **CPU**: 4+ cores para processamento ML
- **RAM**: 16GB para datasets completos
- **Disco**: SSD para melhor performance I/O

## Funcionalidades Avançadas

### Detection Engine
- **Real-time Processing**: Análise de tráfego em tempo real via Kafka
- **Multi-layer Detection**: Detecção em múltiplas camadas de rede
- **Adaptive Thresholds**: Limites que se ajustam ao comportamento normal
- **False Positive Reduction**: Técnicas para minimizar falsos positivos

### Mitigation System
- **Automated Response**: Resposta automática baseada em severidade
- **Cooperative Defense**: Partilha de inteligência entre nós
- **Traffic Shaping**: Modelação inteligente de tráfego
- **Recovery Mechanisms**: Recuperação automática após mitigação

### Monitoring & Analytics
- **Real-time Dashboards**: Visualização em tempo real no Grafana
- **Performance Metrics**: Métricas detalhadas de sistema e ML
- **Alert System**: Sistema de alertas configurável
- **Historical Analysis**: Análise histórica de ataques e performance

## Estrutura do Projeto

```
ddos-mitigation-lab/
├── deployment/scripts/       # Scripts operacionais
│   ├── make.sh              # Comando principal
│   ├── train_*.py           # Scripts de treino ML
│   └── optimize_*.py        # Otimização de modelos
├── src/                     # Código fonte
│   ├── ml-processor/        # Pipeline ML principal
│   ├── bgp-controller/      # Controlo BGP
│   ├── data-ingestion/      # Ingestão de dados
│   ├── datasets/            # Datasets e modelos
│   └── models/              # Modelos treinados
├── setup/linux/             # Scripts de instalação
├── docs/                    # Documentação técnica
└── monitoring/              # Configurações Grafana/Prometheus
```
## Instalação e Configuração

### Método Automático (Recomendado)
```bash
git clone https://github.com/pakotes/ddos-mitigation-lab.git
cd ddos-mitigation-lab/setup/linux
chmod +x setup-complete.sh
./setup-complete.sh
```

### Download de Datasets
O sistema pode operar com dados sintéticos ou datasets reais:

```bash
# Download automático do CIC-DDoS2019
./setup/linux/setup_datasets.sh auto

# Download manual disponível em:
# http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/
```

## Sistemas Suportados

- **Rocky Linux 10.0+** (Recomendado)
- **CentOS 8+** 
- **RHEL 8+**
- **Ubuntu 22.04** (Suporte experimental)

## Troubleshooting

### Problemas Comuns

**Docker não inicia:**
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
# Reiniciar sessão
```

**Portas ocupadas:**
```bash
./deployment/scripts/make.sh clean
sudo netstat -tulpn | grep :3000
```

**Falta de espaço:**
```bash
docker system prune -a
df -h  # Verificar espaço disponível
```

## Documentação Adicional

- **[Arquitetura](docs/architecture.md)** - Documentação técnica detalhada
- **[Datasets](docs/dataset-comparison.md)** - Informação sobre datasets utilizados
- **[Hierarquia de Modelos](docs/model-hierarchy.md)** - Sistema de carregamento de modelos
- **[Troubleshooting](docs/troubleshooting/)** - Resolução de problemas específicos

## Performance e Benchmarks

### Métricas de Detecção
- **Precisão**: >95% em datasets de teste
- **Recall**: >92% para ataques conhecidos
- **Latência**: <200ms para detecção em tempo real
- **Throughput**: 10K+ pacotes/segundo

### Recursos Necessários
- **CPU**: 2-4 cores durante operação normal
- **RAM**: 2-4GB durante detecção, 8-16GB durante treino
- **I/O**: Dependente do volume de tráfego analisado

## Contribuição e Desenvolvimento

### Estrutura de Desenvolvimento
```bash
# Ambiente de desenvolvimento
git clone https://github.com/pakotes/ddos-mitigation-lab.git
cd ddos-mitigation-lab

# Testes
./deployment/scripts/make.sh test

# Logs de desenvolvimento
./deployment/scripts/make.sh logs
```

### Extensibilidade
O sistema foi desenhado para ser facilmente extensível:

- **Novos Algoritmos ML**: Adicionar em `src/ml-processor/`
- **Fontes de Dados**: Integrar via `src/data-ingestion/`
- **Estratégias de Mitigação**: Estender `src/bgp-controller/`
- **Dashboards**: Personalizar em `monitoring/grafana/`

## Licença e Suporte

**Licença**: MIT License  
**Status**: Projeto ativo e mantido  
**Suporte**: Issues no GitHub para questões técnicas

---

Para começar rapidamente:
```bash
curl -fsSL https://raw.githubusercontent.com/pakotes/ddos-mitigation-lab/master/setup/linux/setup-complete.sh | bash
cd ddos-mitigation-lab && ./deployment/scripts/make.sh up
```

Acesse http://localhost:3000 para ver os dashboards em ação.
