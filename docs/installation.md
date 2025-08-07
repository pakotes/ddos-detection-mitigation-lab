# Installation Guide - DDoS Detection & Mitigation Lab

## Quick Start Installation

### Linux (Rocky Linux - Recommended)
```bash
# Automated complete setup
curl -O https://raw.githubusercontent.com/pakotes/ddos-mitigation-lab/master/setup/linux/rocky-linux/setup-rocky-linux.sh
chmod +x setup-rocky-linux.sh
sudo ./setup-rocky-linux.sh

# Clone and start system
git clone https://github.com/pakotes/ddos-mitigation-lab.git
cd ddos-mitigation-lab
./deployment/scripts/make.sh start
```

## System Requirements

### Operating System Support
- **Linux**: Rocky Linux 10.0 (recommended), Ubuntu 20.04+, CentOS 8+
- **Windows**: Windows 10+ with Docker Desktop
- **macOS**: macOS 11+ with Docker Desktop

### Minimum Hardware Requirements
- **RAM**: 8GB
- **CPU**: 4 cores
- **Storage**: 50GB available space
- **Network**: Internet connection

### Recommended Hardware Configuration
- **RAM**: 16GB+
- **CPU**: 8+ cores (for optimal ML performance)
- **Storage**: 100GB+ SSD (for dataset processing)
- **Network**: High-speed internet connection

## Essential Commands

```bash
# System status
./deployment/scripts/make.sh status

# Stop system
./deployment/scripts/make.sh down

# View logs
./deployment/scripts/make.sh logs

# Rebuild containers
./deployment/scripts/make.sh rebuild
```

## Web Interfaces

After successful initialization:

- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Prometheus Metrics**: http://localhost:9090
- **ML Processor API**: http://localhost:8000
- **BGP Controller**: http://localhost:8001
- **Data Ingestion Service**: http://localhost:8002

## Troubleshooting

### Docker Service Issues
```bash
# Restart Docker service (Linux)
sudo systemctl restart docker

# Windows: Restart Docker Desktop application
```

### Port Conflicts
```bash
# Check ports in use
netstat -tulpn | grep :3000

# Stop conflicting containers
docker stop $(docker ps -q)
```

### Permission Errors (Linux)
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again for changes to take effect
```

### Debug Logging
```bash
# View all service logs
./deployment/scripts/make.sh logs

# Specific service logs
docker logs ml-processor
docker logs bgp-controller
docker logs data-ingestion
```

### Common Installation Issues

#### Dataset Download Failures
- **Cause**: Network connectivity or storage space
- **Solution**: Check internet connection and ensure 50GB+ available space

#### ML Model Loading Errors
- **Cause**: Missing model files or insufficient memory
- **Solution**: Run setup scripts to download models, ensure 8GB+ RAM

#### BGP Controller Startup Failures
- **Cause**: Port conflicts or missing dependencies
- **Solution**: Check port 8001 availability, verify ExaBGP installation

## Performance Optimization

### Resource Allocation
```yaml
# Docker Compose resource limits
services:
  ml-processor:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

### Storage Optimization
```bash
# Clean up unused Docker resources
docker system prune -f

# Remove old model files
rm -rf src/models/old_*
```

## Additional Documentation

- **System Architecture**: [architecture.md](architecture.md)
- **Advanced Setup**: [setup/README.md](../setup/README.md)
- **Dataset Analysis**: [dataset-comparison.md](dataset-comparison.md)
- **Troubleshooting Guide**: [troubleshooting/README.md](troubleshooting/README.md)

## Verification Steps

### System Health Check
```bash
# Check all services are running
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test ML processor endpoint
curl http://localhost:8000/health

# Verify Prometheus metrics
curl http://localhost:9090/api/v1/query?query=up
```

### Performance Validation
```bash
# Run benchmark test
./deployment/scripts/benchmark.py

# Check detection latency
./deployment/scripts/analyze_results.py --latency
```
