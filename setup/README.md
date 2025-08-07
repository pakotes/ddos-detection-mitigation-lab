# Setup e Instalação

Sistema de instalação automatizada para ambiente Linux e preparação de datasets.

## Estrutura

```
setup/
├── linux/                    # Scripts de instalação Linux
├── windows/                  # Scripts de instalação Windows
├── dataset-preparation/      # Sistema de preparação de datasets
└── README.md                # Esta documentação
```

## Instalação Rápida

### Método Recomendado
```bash
git clone https://github.com/pakotes/ddos-mitigation-lab.git
cd ddos-mitigation-lab
chmod +x first-run.sh
./first-run.sh
```

### Método Manual
```bash
git clone https://github.com/pakotes/ddos-mitigation-lab.git
cd ddos-mitigation-lab/setup
chmod +x install.sh
./install.sh
```

## Preparação de Datasets

Após a instalação do sistema, prepare os datasets de segurança:

```bash
cd setup/dataset-preparation
python prepare_datasets.py
```

Ver documentação completa em `setup/dataset-preparation/README.md`

## Scripts Disponíveis

### `install.sh` - Instalação Completa
Setup automático de todo o sistema Linux.

```bash
./setup/install.sh
```

**O que instala:**
- Docker + Docker Compose
- Python 3 + pip + dependências ML
- Estrutura de diretórios
- Permissões e configurações

### `datasets.sh` - Download de Datasets
Download e processamento de datasets de treino.

```bash
./setup/datasets.sh
```

**Datasets suportados:**
- **CIC-DDoS2019**: Dataset especializado em ataques DDoS
- **UNSW-NB15**: Dataset geral de segurança de rede  
- **Sintéticos**: Fallback para desenvolvimento

## Uso Típico

```bash
# 1. Clone e setup inicial
git clone https://github.com/pakotes/ddos-mitigation-lab.git
cd ddos-mitigation-lab
./setup/install.sh

# 2. Download de datasets
./setup/datasets.sh

# 3. Iniciar sistema
./deployment/scripts/make.sh up
```

## Simplificação da Estrutura

Esta estrutura simplificada substitui os 6 scripts complexos anteriores por 2 scripts essenciais.

**Scripts removidos:**
- `setup-complete.sh` (634 linhas)
- `setup_datasets.sh` (1211 linhas)  
- `download-cicddos2019.sh` (358 linhas)
- `setup-curl.sh` (132 linhas)
- `rocky-linux/setup-rocky-linux.sh`
- Arquivos de backup corrompidos

**Novos scripts:**
- `install.sh` (150 linhas) - Instalação do sistema
- `datasets.sh` (200 linhas) - Download de datasets

**Resultado:** Redução de 2335+ linhas para 350 linhas (85% de redução)

## Componentes Instalados

### Sistema Base
- **Docker & Docker Compose**: Instalação oficial via repositórios
- **Python 3.8+**: Ambiente Python com pip atualizado
- **Ferramentas de sistema**: git, curl, wget, unzip, htop

### Dependências ML
- **NumPy, Pandas**: Processamento de dados
- **XGBoost**: Algoritmo de gradient boosting
- **scikit-learn**: Framework ML principal
- **Isolation Forest**: Detecção de anomalias

### Configurações do Sistema
- **Limites de recursos**: Configuração de ulimits
- **Parâmetros sysctl**: Otimizações de rede
- **Firewall**: Configuração básica de portas
- **Aliases convenientes**: Comandos ddos-* para operação

## Datasets Suportados

### CIC-DDoS2019 (Recomendado)
- **Fonte**: Canadian Institute for Cybersecurity
- **URL**: http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/
- **Tamanho**: ~1.5GB comprimido
- **Tipos de ataque**: 12 variantes de DDoS

### UNSW-NB15 v3 (Complementar)
- **Uso**: Detecção de anomalias gerais
- **Integração**: Feature engineering automatizado
- **Formato**: Dados pré-processados

## Sistemas Operacionais Suportados

| Sistema | Versão | Status | Comando |
|---------|--------|--------|---------|
| Rocky Linux | 10.0+ | Recomendado | `./install.sh` |
| CentOS | 8+ | Testado | `./install.sh` |
| RHEL | 8+ | Compatível | `./install.sh` |
| Ubuntu | 22.04+ | Testado | `./install.sh` |
| Debian | 11+ | Compatível | `./install.sh` |

## Requisitos do Sistema

### Mínimos
- **CPU**: 2 cores
- **RAM**: 4GB
- **Disco**: 10GB livres
- **Rede**: Conexão com internet
- **Privilégios**: sudo para instalação

### Recomendados
- **CPU**: 4+ cores para ML
- **RAM**: 8GB para datasets completos
- **Disco**: 20GB para desenvolvimento
- **SSD**: Para melhor performance I/O

## Processo de Instalação

### 1. Verificação de Pré-requisitos
```bash
# Verificação automática durante setup
- Sistema operacional suportado
- Acesso sudo disponível
- Conectividade de rede
- Espaço em disco suficiente
```

### 2. Instalação de Dependências
```bash
# Pacotes do sistema
sudo dnf install -y docker docker-compose python3 python3-pip git

# Dependências Python
pip3 install numpy pandas scikit-learn xgboost
```

### 3. Configuração de Serviços
```bash
# Docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# Configurações de sistema
echo "* soft nofile 65535" >> /etc/security/limits.conf
```

### 4. Download de Datasets
```bash
# Automático durante setup
./datasets.sh

# Dataset específico
./datasets.sh cicddos2019
```

## Opções de Instalação

### Instalação Completa
```bash
./install.sh
```
Instala tudo: sistema, dependências, configurações

### Instalação Seletiva
```bash
# Apenas datasets
./datasets.sh

# Dataset específico
./datasets.sh unsw
```

### Instalação Remota
```bash
# Via curl
curl -fsSL https://raw.githubusercontent.com/pakotes/ddos-mitigation-lab/master/setup/install.sh | bash
```

## Verificação Pós-Instalação

### Comandos de Teste
```bash
# Docker funcionando
docker --version
docker compose version

# Python e dependências
python3 -c "import numpy, pandas, sklearn, xgboost; print('ML libs OK')"

# Projeto funcional
cd ddos-mitigation-lab
./deployment/scripts/make.sh test
```

### Aliases Criados
```bash
ddos-up          # Iniciar sistema
ddos-down        # Parar sistema
ddos-status      # Status containers
ddos-logs        # Logs em tempo real
ddos-cd          # Navegar para projeto
ddos-check       # Verificar sistema
```

## Troubleshooting

### Docker não funciona
```bash
# Verificar serviço
sudo systemctl status docker

# Reiniciar se necessário
sudo systemctl restart docker

# Verificar grupos do usuário
groups $USER | grep docker
```

### Python/pip problemas
```bash
# Verificar instalação
python3 --version
pip3 --version

# Reinstalar se necessário
sudo dnf reinstall python3-pip
```

### Datasets não baixam
```bash
# Verificar conectividade
curl -I http://cicresearch.ca

# Download manual
wget http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/...

# Verificar espaço
df -h
```

### Permissões negadas
```bash
# Verificar sudo
sudo -l

# Relogar após instalação
logout
# login novamente
```

## Desinstalação

### Limpeza Completa
```bash
# Parar containers
./deployment/scripts/make.sh down

# Remover containers e imagens
docker system prune -a

# Remover aliases (manual)
# Editar ~/.bashrc e remover linhas ddos-*
```

### Limpeza Seletiva
```bash
# Apenas containers do projeto
./deployment/scripts/make.sh clean

# Apenas datasets
rm -rf src/datasets/integrated/
```

## Suporte e Documentação

### Logs de Instalação
- Logs salvos em `/tmp/ddos-setup.log`
- Modo verbose disponível com `bash -x install.sh`

### Documentação Adicional
- **[README principal](../README.md)** - Visão geral do projeto
- **[Arquitetura](../docs/architecture.md)** - Detalhes técnicos
- **[Troubleshooting](../docs/troubleshooting/)** - Problemas específicos

### Suporte
- Issues no GitHub para problemas técnicos
- Logs detalhados para diagnóstico
- Scripts de verificação incluídos