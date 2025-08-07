# 🪟 Troubleshooting Windows - DDoS Mitigation Lab

Resolução de problemas específicos para sistemas Windows.

## 📋 Versões Testadas

- ✅ Windows 10 (versão 1909+)
- ✅ Windows 11
- 🔄 Windows Server 2019/2022 (em teste)

## 🚨 Problemas Críticos

### PowerShell Execution Policy

#### Erro: "execution of scripts is disabled"
```powershell
# Verificar política atual
Get-ExecutionPolicy

# Mostrar todas as políticas
Get-ExecutionPolicy -List

# Permitir execução para usuário atual
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Ou executar com bypass (uma vez)
PowerShell -ExecutionPolicy Bypass -File .\setup\windows\setup_datasets_auto.ps1
```

#### Scripts não executam mesmo com política alterada
```powershell
# Desbloquear arquivo se veio da internet
Unblock-File -Path .\setup\windows\setup_datasets_auto.ps1

# Executar como Administrador
Start-Process PowerShell -Verb RunAs -ArgumentList "-ExecutionPolicy Bypass -File .\setup\windows\setup_datasets_auto.ps1"
```

## 🐳 Docker Desktop Issues

### Docker Desktop não inicia

#### Verificar pré-requisitos
```powershell
# Verificar se virtualização está habilitada
Get-ComputerInfo | Select-Object HyperVRequirementVirtualizationFirmwareEnabled

# Verificar Hyper-V
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V

# Habilitar Hyper-V se necessário (requer reinicialização)
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
```

#### Problemas com WSL 2
```powershell
# Verificar versão WSL
wsl --version

# Atualizar WSL
wsl --update

# Verificar distribuições instaladas
wsl --list --verbose

# Definir versão padrão
wsl --set-default-version 2
```

### Docker não responde
```powershell
# Reiniciar Docker Desktop
Stop-Process -Name "Docker Desktop" -Force
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Ou via serviços
Restart-Service -Name com.docker.service
```

### Erro: "Hardware assisted virtualization"
1. Entrar no BIOS/UEFI
2. Habilitar:
   - Intel: VT-x
   - AMD: AMD-V
3. Habilitar virtualização aninhada se em VM

## 📁 Problemas de Arquivo e Permissão

### Arquivos ZIP não extraem
```powershell
# Verificar se arquivo existe e não está corrompido
Test-Path ".\datasets\cicddos2019.zip"
Get-FileHash ".\datasets\cicddos2019.zip" -Algorithm SHA256

# Tentar extrair manualmente
Expand-Archive -Path ".\datasets\cicddos2019.zip" -DestinationPath ".\datasets\cicddos2019" -Force
```

### Problemas de permissão em pastas
```powershell
# Executar como Administrador
Start-Process PowerShell -Verb RunAs

# Verificar permissões
Get-Acl ".\datasets" | Format-List

# Ajustar permissões (como Admin)
icacls ".\datasets" /grant "Users:(OI)(CI)F" /T
```

### Path muito longo
```powershell
# Habilitar caminhos longos no Windows
# Como Administrador:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Ou usar paths UNC
$longPath = "\\?\C:\Projectos\ProjetoMestrado\ddos-mitigation-lab\datasets\very\long\path"
```

## 🌐 Problemas de Rede

### Porta já em uso
```powershell
# Verificar qual processo usa a porta
Get-NetTCPConnection -LocalPort 8080

# Obter detalhes do processo
Get-Process -Id (Get-NetTCPConnection -LocalPort 8080).OwningProcess

# Matar processo específico
Stop-Process -Id <PID> -Force
```

### Windows Defender bloqueia containers
```powershell
# Adicionar exclusão para pasta do projeto
Add-MpPreference -ExclusionPath "C:\Projectos\ProjetoMestrado\ddos-mitigation-lab"

# Adicionar exclusão para Docker
Add-MpPreference -ExclusionPath "C:\ProgramData\Docker"
Add-MpPreference -ExclusionPath "C:\Program Files\Docker"
```

### Firewall bloqueando portas
```powershell
# Verificar regras do Windows Firewall
Get-NetFirewallRule | Where-Object {$_.Direction -eq "Inbound" -and $_.Action -eq "Allow"}

# Adicionar regra para porta específica
New-NetFirewallRule -DisplayName "DDoS Lab Grafana" -Direction Inbound -Protocol TCP -LocalPort 3000 -Action Allow

# Adicionar regra para range de portas
New-NetFirewallRule -DisplayName "DDoS Lab Ports" -Direction Inbound -Protocol TCP -LocalPort 8000-9999 -Action Allow
```

## 💾 Problemas de Armazenamento

### Espaço insuficiente
```powershell
# Verificar espaço disponível
Get-WmiObject -Class Win32_LogicalDisk | Select-Object DeviceID, Size, FreeSpace

# Limpar arquivos temporários
Remove-Item -Path $env:TEMP\* -Recurse -Force -ErrorAction SilentlyContinue

# Limpar Docker
docker system prune -a
```

### Drive não compartilhado no Docker
1. Abrir Docker Desktop
2. Settings > Resources > File Sharing
3. Adicionar drive C:\ (ou drive do projeto)
4. Apply & Restart

## 🔧 Problemas com Scripts PowerShell

### Módulos não encontrados
```powershell
# Verificar módulos instalados
Get-Module -ListAvailable

# Instalar módulo específico
Install-Module -Name <ModuleName> -Force

# Atualizar PowerShell Get
Install-Module PowerShellGet -Force
```

### Encoding de arquivo
```powershell
# Se scripts têm caracteres estranhos
# Salvar arquivo como UTF-8 with BOM no VSCode ou Notepad++

# Converter encoding
Get-Content .\script.ps1 -Encoding UTF8 | Set-Content .\script_fixed.ps1 -Encoding UTF8
```

### Variáveis de ambiente
```powershell
# Verificar PATH
$env:PATH -split ';'

# Verificar variáveis Docker
$env:DOCKER_HOST
$env:COMPOSE_PROJECT_NAME

# Definir variável temporariamente
$env:COMPOSE_PROJECT_NAME = "ddos-mitigation-lab"
```

## 🖥️ Performance e Recursos

### Sistema lento/travando
```powershell
# Verificar uso de CPU e memória
Get-Process | Sort-Object CPU -Descending | Select-Object -First 10
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10

# Verificar uso do Docker
docker stats

# Limitar recursos do Docker Desktop
# Settings > Resources > Advanced
# Ajustar CPU cores e Memory
```

### WSL consumindo muita memória
```powershell
# Criar arquivo .wslconfig no home do usuário
@"
[wsl2]
memory=4GB
processors=2
swap=2GB
"@ | Out-File -FilePath "$env:USERPROFILE\.wslconfig" -Encoding utf8

# Reiniciar WSL
wsl --shutdown
```

## 📱 Problemas com Antivírus

### Antivírus bloqueia execução
```powershell
# Adicionar exclusões no Windows Defender
Add-MpPreference -ExclusionPath "C:\Projectos\ProjetoMestrado\ddos-mitigation-lab"
Add-MpPreference -ExclusionProcess "docker.exe"
Add-MpPreference -ExclusionProcess "dockerd.exe"

# Para outros antivírus, consultar documentação específica
```

## 🔍 Debug e Logs

### Coletar informações do sistema
```powershell
# Informações básicas
Get-ComputerInfo | Out-File system_info.txt

# Versões importantes
docker version | Out-File -Append system_info.txt
Get-Host | Out-File -Append system_info.txt

# Logs do Docker
docker-compose logs | Out-File docker_logs.txt

# Logs do Windows
Get-WinEvent -LogName System -MaxEvents 50 | Out-File windows_logs.txt
```

### Debug de container específico
```powershell
# Logs detalhados
docker logs --details ml-processor

# Entrar no container (se possível)
docker exec -it ml-processor cmd
# ou
docker exec -it ml-processor powershell
```

## 🆘 Casos Extremos

### Reset completo do Docker
```powershell
# CUIDADO: Remove todos os containers, imagens e volumes
docker system prune -a --volumes

# Reset completo do Docker Desktop
# 1. Desinstalar Docker Desktop
# 2. Remover pastas:
Remove-Item -Path "$env:APPDATA\Docker" -Recurse -Force
Remove-Item -Path "$env:LOCALAPPDATA\Docker" -Recurse -Force
Remove-Item -Path "C:\ProgramData\Docker" -Recurse -Force
# 3. Reinstalar Docker Desktop
```

### Reset do WSL
```powershell
# Backup de dados importantes antes!
wsl --unregister Ubuntu
wsl --install Ubuntu
```

## 📞 Suporte

Para obter ajuda adicional:

1. **Coletar informações do sistema**:
   ```powershell
   # Executar e salvar resultado
   .\scripts\collect_debug_windows.ps1 > debug_info.txt
   ```

2. **Informações essenciais**:
   ```powershell
   Write-Host "=== Sistema ==="
   Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, TotalPhysicalMemory
   Write-Host "=== Docker ==="
   docker version
   docker-compose version
   Write-Host "=== PowerShell ==="
   $PSVersionTable
   ```

3. **Screenshots dos erros** quando possível

4. **Criar issue no GitHub** com todas as informações coletadas
