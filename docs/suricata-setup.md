# Suricata Network Detection Setup

Suricata is installed directly on the host system for maximum packet capture performance and network visibility.

## Installation

### Ubuntu/Debian Systems
```bash
sudo add-apt-repository ppa:oisf/suricata-stable
sudo apt update && sudo apt install suricata
```

### CentOS/RHEL/Rocky Linux
```bash
sudo dnf install epel-release
sudo dnf install suricata
```

### Manual Installation (Latest Version)
```bash
# Download and compile latest version
wget https://www.openinfosecfoundation.org/download/suricata-7.0.0.tar.gz
tar xzf suricata-7.0.0.tar.gz
cd suricata-7.0.0
./configure --enable-nfqueue --prefix=/usr --sysconfdir=/etc --localstatedir=/var
make && sudo make install
```

## Configuration

### 1. Configure EVE JSON Output
```bash
sudo nano /etc/suricata/suricata.yaml
```

Configure structured JSON logging:
```yaml
outputs:
  - eve-log:
      enabled: yes
      filetype: regular
      filename: /var/log/suricata/eve.json
      types:
        - alert:
            payload: yes
            payload-buffer-size: 4kb
            payload-printable: yes
            packet: yes
        - http:
            extended: yes
        - dns:
            query: yes
            answer: yes
        - flow:
            memcap: 128mb
        - netflow
        - stats:
            totals: yes
            threads: no
            deltas: yes
```

### 2. Network Interface Configuration
```bash
# Configure interface for monitoring
sudo nano /etc/suricata/suricata.yaml
```

Set monitoring interface:
```yaml
af-packet:
  - interface: eth0
    cluster-id: 99
    cluster-type: cluster_flow
    defrag: yes
  - interface: default
```

### 3. Rule Management
```bash
# Update rule sets
sudo suricata-update

# Enable specific rule categories
sudo suricata-update enable-source et/open
sudo suricata-update enable-source oisf/trafficid
```

### 4. Service Management
```bash
# Start Suricata daemon
sudo suricata -c /etc/suricata/suricata.yaml -i eth0 -D

# Or use systemd service
sudo systemctl enable suricata
sudo systemctl start suricata
sudo systemctl status suricata
```

### 5. Log Verification
```bash
# Monitor real-time events
sudo tail -f /var/log/suricata/eve.json

# Check for specific event types
sudo grep '"event_type":"alert"' /var/log/suricata/eve.json | jq .
```

### 6. Docker Integration Permissions
```bash
# Set appropriate permissions for Docker access
sudo chmod 644 /var/log/suricata/eve.json
sudo chown root:docker /var/log/suricata/
sudo usermod -aG docker suricata
```

## Performance Optimization

### High-Performance Configuration
```yaml
# In /etc/suricata/suricata.yaml
threading:
  set-cpu-affinity: yes
  cpu-affinity:
    - management-cpu-set:
        cpu: [ 0 ]
    - receive-cpu-set:
        cpu: [ 1 ]
    - worker-cpu-set:
        cpu: [ 2, 3, 4, 5 ]

af-packet:
  - interface: eth0
    threads: 4
    cluster-id: 99
    cluster-type: cluster_flow
    ring-size: 8192
    block-size: 32768
```

### Memory Optimization
```yaml
# Adjust memory limits
flow:
  memcap: 256mb
  hash-size: 65536
  prealloc: 10000

stream:
  memcap: 256mb
  checksum-validation: yes
  reassembly:
    memcap: 256mb
    depth: 1mb
```

## Development and Testing

### Synthetic Data Generation
```bash
# Use built-in test data generation
./deployment/scripts/make.sh generate-test-data

# Or manually generate network events
sudo tcpreplay -i eth0 test-data/sample.pcap
```

### Local Testing Setup
```bash
# Start Suricata in testing mode
sudo suricata -c /etc/suricata/suricata.yaml -r test-data/sample.pcap
```

## Monitoring and Troubleshooting

### Service Status Check
```bash
# Check Suricata service status
sudo systemctl status suricata

# View service logs
sudo journalctl -u suricata -f

# Check Suricata internal logs
sudo tail -f /var/log/suricata/suricata.log
```

### Network Interface Issues
```bash
# List available interfaces
ip link show

# Enable promiscuous mode
sudo ip link set eth0 promisc on

# Check interface statistics
sudo suricata --list-app-layer-protos
```

### Performance Monitoring
```bash
# Monitor Suricata statistics
sudo suricata-update list-sources

# Check rule loading
sudo suricata -T -c /etc/suricata/suricata.yaml

# View detection statistics
sudo grep '"event_type":"stats"' /var/log/suricata/eve.json | tail -1 | jq .
```

### Common Issues and Solutions

#### No Events Generated
- **Check interface configuration**: Verify correct network interface
- **Rule validation**: Ensure rules are loaded correctly
- **Permissions**: Check file permissions for log directory

#### High CPU Usage
- **Thread allocation**: Optimize CPU affinity settings
- **Rule optimization**: Disable unnecessary rule sets
- **Buffer sizing**: Adjust ring and block sizes

#### Log Rotation
```bash
# Configure logrotate for Suricata logs
sudo nano /etc/logrotate.d/suricata

/var/log/suricata/*.log /var/log/suricata/*.json {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 644 suricata suricata
    postrotate
        /bin/kill -HUP $(cat /var/run/suricata.pid 2>/dev/null) 2>/dev/null || true
    endscript
}
```

## Integration with ML Pipeline

The generated `eve.json` logs are consumed by the Data Ingestion service, which normalizes the data and forwards it to the ML processing pipeline for real-time threat detection.
