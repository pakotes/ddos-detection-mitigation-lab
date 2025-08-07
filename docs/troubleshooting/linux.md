# 🐧 Troubleshooting Linux - DDoS Mitigation Lab

Resolução de problemas específicos para sistemas Linux.

## 📋 Distribuições Testadas

- ✅ Rocky Linux 10.0
- ✅ CentOS 8+
- 🔄 Ubuntu 22.04 LTS (em teste)
- 🔄 Debian 12 (em teste)

## 🚨 Problemas Críticos

### Script de instalação falha

#### Erro: "Permission denied"
```bash
# Verificar se tem privilégios sudo
sudo whoami

# Tornar script executável
chmod +x setup/linux/rocky-linux/setup-rocky-linux.sh

# Executar com sudo
sudo ./setup/linux/rocky-linux/setup-rocky-linux.sh
```

#### Erro: "Package not found"
```bash
# Atualizar repositórios
sudo dnf clean all
sudo dnf update

# Verificar se EPEL está habilitado
sudo dnf install epel-release

# Reexecutar instalação
sudo ./setup/linux/rocky-linux/setup-rocky-linux.sh
```

## 🐳 Docker Issues

### Docker daemon não inicia
```bash
# Verificar status
sudo systemctl status docker

# Verificar logs
sudo journalctl -u docker

# Reiniciar serviço
sudo systemctl restart docker

# Habilitar na inicialização
sudo systemctl enable docker
```

### Usuário não consegue usar Docker
```bash
# Verificar grupos do usuário
groups $USER

# Adicionar ao grupo docker
sudo usermod -aG docker $USER

# IMPORTANTE: Fazer logout e login novamente
# Ou usar newgrp para ativar grupo imediatamente
newgrp docker

# Testar sem sudo
docker run hello-world
```

### Container não consegue bind nas portas
```bash
# Verificar se porta está em uso
sudo netstat -tulpn | grep :8080

# Verificar processos
sudo lsof -i :8080

# Matar processo se necessário
sudo kill -9 <PID>
```

## 🔥 Firewall (firewalld)

### Portas bloqueadas
```bash
# Verificar zonas ativas
sudo firewall-cmd --get-active-zones

# Verificar regras da zona padrão
sudo firewall-cmd --list-all

# Adicionar porta específica
sudo firewall-cmd --permanent --add-port=3000/tcp
sudo firewall-cmd --permanent --add-port=9090/tcp
sudo firewall-cmd --permanent --add-port=8080/tcp

# Recarregar configuração
sudo firewall-cmd --reload

# Verificar se foi aplicado
sudo firewall-cmd --list-ports
```

### Abrir range de portas
```bash
# Para todas as portas do lab (8000-9999)
sudo firewall-cmd --permanent --add-port=8000-9999/tcp
sudo firewall-cmd --reload
```

### Desabilitar firewall temporariamente (não recomendado)
```bash
# Parar firewall
sudo systemctl stop firewalld

# Desabilitar na inicialização
sudo systemctl disable firewalld

# Para reabilitar
sudo systemctl enable firewalld
sudo systemctl start firewalld
```

## 🛡️ SELinux

### Verificar status
```bash
# Status atual
getenforce

# Histórico de negações
sudo ausearch -m AVC -ts recent
```

### Problemas de context
```bash
# Restaurar context padrão
sudo restorecon -Rv /path/to/ddos-mitigation-lab/

# Verificar context atual
ls -Z ./datasets

# Definir context para containers
sudo setsebool -P container_manage_cgroup true
```

### Desabilitar temporariamente (debug only)
```bash
# Modo permissivo
sudo setenforce 0

# Verificar se resolveu o problema
# IMPORTANTE: Reabilitar após debug
sudo setenforce 1
```

## 💾 Espaço em Disco

### Verificar uso
```bash
# Espaço geral
df -h

# Espaço do Docker
docker system df

# Uso por diretório
du -sh ./datasets/*
```

### Limpeza
```bash
# Limpar containers parados
docker container prune

# Limpar imagens não utilizadas
docker image prune

# Limpar volumes órfãos
docker volume prune

# Limpeza completa (CUIDADO!)
docker system prune -a
```

## 🔧 Permissões de Arquivo

### Datasets inacessíveis
```bash
# Verificar proprietário
ls -la ./datasets/

# Ajustar proprietário
sudo chown -R $USER:$USER ./datasets/

# Ajustar permissões
chmod -R 755 ./datasets/
```

### Scripts não executam
```bash
# Tornar executável
chmod +x setup/linux/*.sh
chmod +x deployment/scripts/*.sh

# Verificar shebang
head -1 setup/linux/setup_datasets.sh
```

## 🌐 Problemas de Rede

### DNS não resolve
```bash
# Testar DNS
nslookup google.com

# Verificar configuração
cat /etc/resolv.conf

# Testar conectividade
ping 8.8.8.8
```

### Containers não se comunicam
```bash
# Verificar rede Docker
docker network ls
docker network inspect ddos-mitigation-lab_default

# Testar conectividade interna
docker exec -it grafana ping prometheus
```

## 🐍 Python Environment

### Módulos não encontrados
```bash
# Verificar versão Python
python3 --version

# Verificar pip
pip3 --version

# Instalar dependências
pip3 install -r requirements.txt

# Verificar instalação
python3 -c "import pandas; print('OK')"
```

### Problemas com venv
```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## 📊 Logs e Debug

### Coletar logs importantes
```bash
# Logs do sistema
sudo journalctl --since "1 hour ago" > system_logs.txt

# Logs do Docker
docker-compose logs > docker_logs.txt

# Logs de um container específico
docker logs ml-processor > ml_logs.txt

# Informações do sistema
uname -a > system_info.txt
docker version >> system_info.txt
```

### Debug de container específico
```bash
# Entrar no container
docker exec -it ml-processor bash

# Verificar processos internos
docker exec -it ml-processor ps aux

# Verificar arquivos
docker exec -it ml-processor ls -la /app/
```

## ⚡ Performance

### Sistema lento
```bash
# Verificar CPU e memória
htop
# ou
top

# Verificar I/O
iotop

# Verificar uso do Docker
docker stats
```

### Otimizações
```bash
# Limitar recursos dos containers
# Editar docker-compose.yml:
# services:
#   ml-processor:
#     mem_limit: 2g
#     cpus: '1.0'

# Usar SSD para datasets grandes
# Mover datasets para SSD se disponível
```

## 📞 Suporte

Se precisar de ajuda adicional:

1. **Coletar informações**:
   ```bash
   curl -s https://raw.githubusercontent.com/pakotes/ddos-mitigation-lab/master/scripts/collect_debug_linux.sh | bash
   ```

2. **Informações do sistema**:
   ```bash
   echo "=== Sistema ===" > debug_info.txt
   cat /etc/os-release >> debug_info.txt
   uname -a >> debug_info.txt
   echo "=== Docker ===" >> debug_info.txt
   docker version >> debug_info.txt
   docker-compose version >> debug_info.txt
   echo "=== Recursos ===" >> debug_info.txt
   free -h >> debug_info.txt
   df -h >> debug_info.txt
   ```

3. **Criar issue no GitHub** com as informações coletadas
