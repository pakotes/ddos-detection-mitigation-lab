# Script de Download de Datasets - DDoS Mitigation Lab
# 
# Este script facilita o download dos datasets académicos necessários:
# - CIC-DDoS2019: Dataset especializado em ataques DDoS 
# - NF-UNSW-NB15-v3: Dataset NetFlow para detecção de intrusões
#
# Autor: DDoS Mitigation Lab
# Data: Agosto 2025

param(
    [switch]$Help,
    [switch]$CheckOnly,
    [switch]$Force
)

# Configurações
$ErrorActionPreference = "Stop"
$DatasetDir = Join-Path $PSScriptRoot "dataset-preparation"
$LogFile = Join-Path $DatasetDir "download.log"

# Cores para output
function Write-Success { param($Message) Write-Host $Message -ForegroundColor Green }
function Write-Warning { param($Message) Write-Host $Message -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host $Message -ForegroundColor Red }
function Write-Info { param($Message) Write-Host $Message -ForegroundColor Cyan }

function Show-Help {
    Write-Host @"
===================================================================
           SCRIPT DE DOWNLOAD DE DATASETS ACADÉMICOS
===================================================================

DESCRIÇÃO:
    Downloads dos datasets oficiais para o projeto DDoS Mitigation Lab

DATASETS INCLUÍDOS:
    • CIC-DDoS2019: Ataques DDoS especializados (13 ficheiros CSV)
    • NF-UNSW-NB15-v3: NetFlow para detecção de intrusões (2 ficheiros CSV)

USO:
    .\setup_datasets.ps1                # Download automático
    .\setup_datasets.ps1 -CheckOnly     # Apenas verificar datasets existentes
    .\setup_datasets.ps1 -Force         # Forçar re-download
    .\setup_datasets.ps1 -Help          # Mostrar esta ajuda

REQUISITOS:
    • PowerShell 5.0+
    • Ligação à Internet
    • ~3GB de espaço livre em disco

LOCALIZAÇÃO DOS DATASETS:
    setup/dataset-preparation/CIC-DDoS2019/
    setup/dataset-preparation/NF-UNSW-NB15-v3/

FONTES OFICIAIS:
    • CIC-DDoS2019: https://www.unb.ca/cic/datasets/ddos-2019.html
    • NF-UNSW-NB15-v3: https://espace.library.uq.edu.au/view/UQ:6e0eda1

===================================================================
"@ -ForegroundColor White
}

function Test-DatasetStructure {
    param($DatasetPath, $ExpectedFiles)
    
    if (-not (Test-Path $DatasetPath)) {
        return $false
    }
    
    $actualFiles = Get-ChildItem -Path $DatasetPath -Recurse -File -Name "*.csv" | Measure-Object | Select-Object -ExpandProperty Count
    return $actualFiles -eq $ExpectedFiles
}

function Show-DatasetStatus {
    Write-Info "=== ESTADO ATUAL DOS DATASETS ==="
    
    # Verificar CIC-DDoS2019
    $cicPath = Join-Path $DatasetDir "CIC-DDoS2019"
    if (Test-DatasetStructure $cicPath 13) {
        Write-Success "✓ CIC-DDoS2019: Completo (13 ficheiros CSV)"
        
        $dir0112 = Join-Path $cicPath "01-12"
        $dir0311 = Join-Path $cicPath "03-11"
        
        if (Test-Path $dir0112) {
            $files0112 = Get-ChildItem -Path $dir0112 -File "*.csv" | Measure-Object | Select-Object -ExpandProperty Count
            Write-Info "  📁 01-12/: $files0112 ficheiros"
        }
        
        if (Test-Path $dir0311) {
            $files0311 = Get-ChildItem -Path $dir0311 -File "*.csv" | Measure-Object | Select-Object -ExpandProperty Count
            Write-Info "  📁 03-11/: $files0311 ficheiros"
        }
    } else {
        Write-Warning "⚠ CIC-DDoS2019: Incompleto ou ausente"
    }
    
    # Verificar NF-UNSW-NB15-v3
    $nfPath = Join-Path $DatasetDir "NF-UNSW-NB15-v3"
    if (Test-DatasetStructure $nfPath 2) {
        Write-Success "✓ NF-UNSW-NB15-v3: Completo (2 ficheiros CSV)"
        
        $featuresFile = Join-Path $nfPath "NetFlow_v3_Features.csv"
        $mainFile = Join-Path $nfPath "NF-UNSW-NB15-v3.csv"
        
        if (Test-Path $featuresFile) {
            $size = [math]::Round((Get-Item $featuresFile).Length / 1MB, 1)
            Write-Info "  📄 NetFlow_v3_Features.csv: ${size}MB"
        }
        
        if (Test-Path $mainFile) {
            $size = [math]::Round((Get-Item $mainFile).Length / 1MB, 1)
            Write-Info "  📄 NF-UNSW-NB15-v3.csv: ${size}MB"
        }
    } else {
        Write-Warning "⚠ NF-UNSW-NB15-v3: Incompleto ou ausente"
    }
    
    Write-Host ""
}

function Show-ManualInstructions {
    Write-Warning @"
===================================================================
              INSTRUÇÕES PARA DOWNLOAD MANUAL
===================================================================

Os datasets académicos requerem download manual devido a políticas
de licenciamento e requisitos de registo académico.

CIC-DDoS2019:
─────────────
1. Aceder: https://www.unb.ca/cic/datasets/ddos-2019.html
2. Preencher formulário de registo académico
3. Descarregar: DDoS2019.zip (~2.3 GB)
4. Extrair para: setup/dataset-preparation/CIC-DDoS2019/
5. Verificar: Devem existir directórios 01-12/ e 03-11/

Alternativa Kaggle:
• https://www.kaggle.com/datasets/aymenabb/ddos-evaluation-dataset-cic-ddos2019

NF-UNSW-NB15-v3:
────────────────
1. Aceder: https://espace.library.uq.edu.au/view/UQ:6e0eda1
2. Descarregar: NF-UNSW-NB15-v3.zip (~800 MB)
3. Extrair para: setup/dataset-preparation/NF-UNSW-NB15-v3/
4. Verificar: Devem existir 2 ficheiros CSV

Alternativa Kaggle:
• https://www.kaggle.com/datasets/ndayisabae/nf-unsw-nb15-v3

VERIFICAÇÃO:
Execute novamente este script após os downloads para verificar
a estrutura dos datasets.

===================================================================
"@
}

function Test-Prerequisites {
    Write-Info "A verificar pré-requisitos..."
    
    # Verificar PowerShell version
    if ($PSVersionTable.PSVersion.Major -lt 5) {
        Write-Error "PowerShell 5.0 ou superior é necessário"
        return $false
    }
    
    # Criar diretório se não existir
    if (-not (Test-Path $DatasetDir)) {
        Write-Info "A criar directório: $DatasetDir"
        New-Item -ItemType Directory -Path $DatasetDir -Force | Out-Null
    }
    
    # Verificar espaço em disco
    $drive = (Get-Item $DatasetDir).PSDrive
    $freeSpace = [math]::Round($drive.Free / 1GB, 2)
    
    if ($freeSpace -lt 5) {
        Write-Warning "Aviso: Apenas ${freeSpace}GB livres. Recomendado: 5GB+"
    } else {
        Write-Success "✓ Espaço em disco suficiente: ${freeSpace}GB"
    }
    
    return $true
}

# Função principal
function Main {
    if ($Help) {
        Show-Help
        return
    }
    
    Write-Host @"
╔══════════════════════════════════════════════════════════════╗
║               DOWNLOAD DE DATASETS ACADÉMICOS               ║
║                    DDoS Mitigation Lab                      ║
╚══════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan
    
    if (-not (Test-Prerequisites)) {
        Write-Error "Pré-requisitos não satisfeitos. A terminar."
        exit 1
    }
    
    Show-DatasetStatus
    
    if ($CheckOnly) {
        Write-Info "Verificação completa. Use o script sem -CheckOnly para instruções de download."
        return
    }
    
    # Verificar se datasets já existem
    $cicComplete = Test-DatasetStructure (Join-Path $DatasetDir "CIC-DDoS2019") 13
    $nfComplete = Test-DatasetStructure (Join-Path $DatasetDir "NF-UNSW-NB15-v3") 2
    
    if ($cicComplete -and $nfComplete -and -not $Force) {
        Write-Success @"

✓ Todos os datasets estão completos!

Para verificar o sistema de processamento:
  cd setup/dataset-preparation
  python prepare_datasets.py --check

Para forçar re-download:
  .\setup_datasets.ps1 -Force
"@
        return
    }
    
    Write-Info @"

Os datasets académicos requerem download manual devido a:
• Políticas de licenciamento institucional
• Requisitos de registo académico
• Termos de uso específicos

"@
    
    Show-ManualInstructions
    
    Write-Info @"

PRÓXIMOS PASSOS:
1. Execute downloads manuais conforme instruções acima
2. Execute novamente: .\setup_datasets.ps1 -CheckOnly
3. Processe datasets: python prepare_datasets.py

"@
}

# Executar script
try {
    Main
} catch {
    Write-Error "Erro: $($_.Exception.Message)"
    Write-Info "Execute '.\setup_datasets.ps1 -Help' para obter ajuda"
    exit 1
}
