# Common Issues - DDoS Detection & Mitigation Lab

Quick reference guide for the most frequently encountered issues and their immediate solutions.

## Quick Fixes

### Container Startup Failures
```bash
# 1. Verify Docker is running
docker --version

# 2. Check service logs
docker-compose logs <service_name>

# 3. Recreate containers
docker-compose down
docker-compose up --build
```

### Web Interface Not Loading
1. **Check container status**: `docker-compose ps`
2. **Test port connectivity**: `curl http://localhost:3000` (Grafana)
3. **Verify firewall settings**: Open ports 3000, 9090, 8080
4. **Wait for initialization**: Containers may take 1-2 minutes to start

### "Port Already in Use" Error
```bash
# Linux
sudo lsof -i :8080
sudo kill -9 <PID>

# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F
```

## Dataset Issues

### Dataset Not Found
```bash
# Verify dataset structure
ls -la ./src/datasets/

# Re-run setup scripts
# Linux
./setup/linux/setup_datasets.sh auto

# Windows
.\setup\windows\setup_datasets_auto.ps1
```

### CSV corrompido ou vazio
```bash
# Verificar arquivo
head -5 ./datasets/cicddos2019/cicddos2019_dataset.csv
wc -l ./datasets/cicddos2019/cicddos2019_dataset.csv

# Reprocessar dataset
rm -rf ./datasets/cicddos2019/
# Executar setup novamente
```

### Dados sintéticos não geram
```bash
# Verificar Python
python3 --version
pip3 list | grep pandas

# Gerar manualmente
python3 scripts/generate_simple_data.py
```

## 🐳 Docker Issues

### "docker-compose command not found"
```bash
# Linux - instalar compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Windows - instalar Docker Desktop
# Ou usar docker compose (sem hífen)
docker compose up
```

### Containers reiniciando constantemente
```bash
# Verificar recursos
docker stats

# Verificar logs de erro
docker logs --tail 50 ml-processor

# Aumentar memória disponível para Docker
# Docker Desktop > Settings > Resources > Memory
```

### Volume mount falha
```bash
# Linux - verificar permissões
sudo chown -R $USER:$USER ./datasets
chmod -R 755 ./datasets

# Windows - compartilhar drive no Docker Desktop
# Settings > Resources > File Sharing
```

## 🌐 Rede e Conectividade

### Grafana mostra "No data"
1. **Verificar Prometheus**: http://localhost:9090/targets
2. **Verificar métricas**: http://localhost:8001/metrics
3. **Restart containers**:
   ```bash
   docker-compose restart prometheus grafana
   ```

### Prometheus não consegue scrape
```bash
# Verificar configuração
docker exec prometheus cat /etc/prometheus/prometheus.yml

# Verificar conectividade interna
docker exec prometheus wget -O- http://ml-processor:8002/metrics
```

### DNS interno não funciona
```bash
# Verificar rede Docker
docker network ls
docker network inspect ddos-mitigation-lab_default

# Recrear rede
docker-compose down
docker network prune
docker-compose up
```

## 🤖 Machine Learning

### Modelo não treina
```bash
# Verificar dados de entrada
docker exec ml-processor ls -la /app/datasets/

# Verificar logs de treinamento
docker logs ml-processor | grep -i "training\|error"

# Treinar manualmente
docker exec ml-processor python model_trainer.py
```

### Predições inconsistentes
```bash
# Verificar modelo carregado
docker exec ml-processor ls -la /app/models/

# Verificar versão do modelo
docker logs ml-processor | grep -i "model loaded"

# Retreinar se necessário
docker exec ml-processor python model_trainer.py --retrain
```

### Performance baixa do ML
1. **Aumentar CPU/RAM para containers**
2. **Verificar se datasets estão em SSD**
3. **Otimizar batch size**:
   ```bash
   # Editar configuração no ml_pipeline.py
   BATCH_SIZE = 1000  # Reduzir se pouca RAM
   ```

## 🔐 Problemas de Permissão

### "Permission denied" (Linux)
```bash
# Scripts
chmod +x setup/linux/*.sh
chmod +x deployment/scripts/*.sh

# Datasets
sudo chown -R $USER:$USER ./datasets/
chmod -R 755 ./datasets/

# Docker (adicionar usuário ao grupo)
sudo usermod -aG docker $USER
# Logout e login novamente
```

### "Access denied" (Windows)
```powershell
# Executar como Administrador
Start-Process PowerShell -Verb RunAs

# Ou alterar política de execução
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Desbloquear arquivos baixados
Unblock-File -Path .\setup\windows\*.ps1
```

## 💾 Espaço em Disco

### "No space left on device"
```bash
# Verificar espaço
df -h

# Limpar Docker
docker system prune -a
docker volume prune

# Limpar datasets antigos
rm -rf ./datasets/backup/
rm -rf ./datasets/temp/

# Mover datasets para disco maior se necessário
```

### Docker consumindo muito espaço
```bash
# Verificar uso do Docker
docker system df

# Limpeza seletiva
docker container prune  # Containers parados
docker image prune      # Imagens órfãs
docker volume prune     # Volumes não utilizados

# Limpeza completa (CUIDADO!)
docker system prune -a --volumes
```

## ⚡ Performance Geral

### Sistema lento
1. **Verificar recursos**:
   ```bash
   # Linux
   htop
   free -h
   
   # Windows
   Task Manager > Performance
   ```

2. **Otimizar Docker**:
   - Limitar CPU/RAM por container
   - Usar SSD para volumes
   - Fechar containers não utilizados

3. **Otimizar datasets**:
   - Usar dados sintéticos para desenvolvimento
   - Processar datasets em chunks menores

### Containers demoram para iniciar
```bash
# Build mais rápido
docker-compose build --parallel

# Usar cache do Docker
docker-compose up --build --no-recreate

# Pre-pull de imagens
docker-compose pull
```

## 🔄 Reset e Limpeza

### Reset completo do projeto
```bash
# Parar tudo
docker-compose down -v

# Limpar Docker
docker system prune -a --volumes

# Limpar datasets
rm -rf ./datasets/*

# Rebuild completo
docker-compose build --no-cache
docker-compose up
```

### Reset só dos dados
```bash
# Parar containers
docker-compose down

# Manter imagens, remover só volumes
docker-compose down -v

# Reexecutar setup
./setup/linux/setup_datasets.sh auto  # Linux
.\setup\windows\setup_datasets_auto.ps1  # Windows

# Reiniciar
docker-compose up
```

## 📞 Quando Pedir Ajuda

Se nenhuma solução funcionar, colete estas informações:

```bash
# Informações do sistema
uname -a  # Linux
systeminfo  # Windows

# Versões
docker version
docker-compose version

# Status dos containers
docker-compose ps

# Logs recentes
docker-compose logs --tail 100

# Uso de recursos
docker stats --no-stream
```

E crie um issue no GitHub com:
1. Descrição clara do problema
2. Passos para reproduzir
3. Mensagens de erro exatas
4. Informações do sistema coletadas acima
