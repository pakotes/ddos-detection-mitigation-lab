# Troubleshooting Guide - DDoS Detection & Mitigation Lab

Comprehensive troubleshooting guide for resolving common issues with the DDoS Detection & Mitigation Lab system.

## Documentation Index

- [Common Issues](common-issues.md) - Quick solutions for frequent problems
- [Linux Troubleshooting](linux.md) - Linux-specific issue resolution
- [Windows Troubleshooting](windows.md) - Windows-specific issue resolution
- [Docker Issues](#docker-issues) - Container and orchestration problems
- [Network Connectivity](#network-connectivity) - Network and connectivity issues
- [Machine Learning](#machine-learning-issues) - ML pipeline problems

## Common Issues

### Container Startup Failures
```bash
# Check container logs
docker logs <container_name>

# Verify system resources
docker system df
free -h

# Check Docker daemon status
sudo systemctl status docker
```

### Port Conflicts
```bash
# Linux - identify process using port
sudo netstat -tulpn | grep :8080
sudo lsof -i :8080

# Windows - identify process using port
netstat -ano | findstr :8080
```

### Permission Issues
```bash
# Linux - fix file permissions
sudo chown -R $USER:$USER ./datasets
chmod +x setup/linux/*.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Windows - run as Administrator
PowerShell -ExecutionPolicy Bypass -File script.ps1
```

## Docker Issues

### Service Not Starting
```bash
# Check Docker daemon status
sudo systemctl status docker

# Restart Docker service
sudo systemctl restart docker

# Verify user is in docker group
groups $USER

# Add user to docker group (requires logout/login)
sudo usermod -aG docker $USER
```

### Container Build Failures
```bash
# Clear build cache
docker builder prune

# Build without cache
docker-compose build --no-cache

# Check disk space
docker system df
docker system prune -f
```

### Container Communication Issues
```bash
# List Docker networks
docker network ls

# Inspect specific network
docker network inspect ddos-mitigation-lab_default

# Recreate network if necessary
docker-compose down
docker network prune -f
docker-compose up
```

### Volume Mount Problems
```bash
# Check volume permissions
ls -la ./src/datasets/

# Fix permissions for Docker access
sudo chown -R $USER:docker ./src/
chmod -R 755 ./src/
```

## Network Connectivity

### Port Binding Failures
```bash
# Linux - identify process using port
sudo lsof -i :3000
sudo netstat -tulpn | grep :3000

# Kill process if necessary
sudo kill -9 <PID>

# Windows - identify and kill process
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

### Firewall Configuration
```bash
# Linux - configure firewall for required ports
sudo ufw allow 3000/tcp    # Grafana
sudo ufw allow 9090/tcp    # Prometheus
sudo ufw allow 8000/tcp    # ML Processor

# CentOS/Rocky Linux
sudo firewall-cmd --permanent --add-port=3000/tcp
sudo firewall-cmd --permanent --add-port=9090/tcp
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

### DNS Resolution Issues
```bash
# Test connectivity to external services
ping google.com
nslookup github.com

# Check Docker DNS
docker run --rm alpine ping google.com
```

## Machine Learning Issues

### Model Loading Failures
```bash
# Check model files exist
ls -la src/models/

# Verify file permissions
chmod 644 src/models/**/*.pkl

# Check available memory
free -h
```

### Training Script Errors
```bash
# Run training with verbose output
python deployment/scripts/train_hybrid_advanced.py --verbose

# Check Python dependencies
pip list | grep -E "(xgboost|sklearn|numpy|pandas)"

# Monitor system resources during training
htop
```

### Performance Issues
```bash
# Monitor ML processor logs
docker logs ml-processor -f

# Check CPU and memory usage
docker stats

# Verify dataset integrity
python -c "import numpy as np; print(np.load('src/datasets/integrated/X_integrated_real.npy').shape)"
```

## System Resource Issues

### Memory Problems
```bash
# Check system memory usage
free -h
ps aux --sort=-%mem | head

# Check Docker memory usage
docker stats --no-stream

# Increase swap if needed (Linux)
sudo swapon -s
sudo dd if=/dev/zero of=/swapfile bs=1G count=4
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Disk Space Issues
```bash
# Check disk usage
df -h
du -sh ./src/datasets/

# Clean up Docker resources
docker system prune -af
docker volume prune -f

# Remove old model files
find ./src/models/ -name "*.pkl" -mtime +30 -delete
```

## Log Analysis

### Container Logs
```bash
# View specific service logs
docker logs data-ingestion --tail 100 -f
docker logs ml-processor --tail 100 -f
docker logs bgp-controller --tail 100 -f

# Export logs for analysis
docker logs ml-processor > ml-processor.log 2>&1
```

### System Performance Monitoring
```bash
# Monitor real-time system metrics
htop
iotop
nethogs

# Monitor Docker container performance
docker stats
docker events
```

## Service-Specific Issues

### Kafka Connection Problems
```bash
# Test Kafka connectivity
docker exec -it kafka kafka-topics.sh --list --bootstrap-server localhost:9092

# Check topic creation
docker exec -it kafka kafka-topics.sh --describe --topic traffic-logs --bootstrap-server localhost:9092
```

### Redis Connection Issues
```bash
# Test Redis connectivity
docker exec -it redis redis-cli ping

# Monitor Redis operations
docker exec -it redis redis-cli monitor
```

### BGP Controller Issues
```bash
# Check ExaBGP status
docker exec -it bgp-controller ps aux | grep exabgp

# Verify BGP configuration
docker exec -it bgp-controller cat /etc/exabgp/exabgp.conf
```

For platform-specific issues, refer to:
- [Linux-specific troubleshooting](linux.md)
- [Windows-specific troubleshooting](windows.md)
- [Common issues and solutions](common-issues.md)
```bash
# Verificar permissões do diretório
ls -la ./datasets

# No Windows, verificar compartilhamento de drive no Docker Desktop
# Configurações > Resources > File Sharing
```

## 🌐 Rede e Conectividade

### Prometheus não consegue scrape métricas
```bash
# Verificar se serviços estão respondendo
curl http://localhost:8001/metrics
curl http://localhost:8002/metrics

# Verificar configuração do Prometheus
docker exec -it prometheus cat /etc/prometheus/prometheus.yml
```

### Grafana não conecta ao Prometheus
```bash
# Verificar logs do Grafana
docker logs grafana

# Testar conectividade interna
docker exec -it grafana wget -O- http://prometheus:9090/api/v1/status/config
```

### Interface web não carrega
1. Verificar se containers estão rodando: `docker-compose ps`
2. Verificar portas expostas: `docker-compose port grafana 3000`
3. Testar conectividade local: `curl http://localhost:3000`
4. Verificar firewall local

## 🤖 Machine Learning

### Modelo não carrega
```bash
# Verificar se arquivo existe
ls -la ./src/models/

# Verificar logs do ml-processor
docker logs ml-processor

# Testar carregamento manual
docker exec -it ml-processor python -c "import pickle; print('OK')"
```

### Dados não são processados
```bash
# Verificar estrutura dos datasets
ls -la ./datasets/

# Verificar logs de processamento
docker logs data-ingestion

# Testar pipeline manualmente
docker exec -it ml-processor python ml_pipeline.py
```

### Performance baixa
1. Verificar recursos disponíveis: `htop` ou Task Manager
2. Ajustar configuração de memória no Docker
3. Verificar se datasets estão otimizados
4. Considerar usar SSD para datasets grandes

## 📞 Suporte Adicional

Se os problemas persistirem:

1. **Coleta de informações**:
   ```bash
   # Linux
   ./scripts/collect_debug_info.sh
   
   # Windows
   .\scripts\collect_debug_info.ps1
   ```

2. **Logs completos**:
   ```bash
   docker-compose logs > full_logs.txt
   ```

3. **Informações do sistema**:
   ```bash
   # Linux
   uname -a
   docker version
   docker-compose version
   
   # Windows
   systeminfo
   docker version
   ```

4. **Criar issue no GitHub** com:
   - Descrição do problema
   - Passos para reproduzir
   - Logs relevantes
   - Informações do sistema
