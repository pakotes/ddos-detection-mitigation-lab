# ML Models

Este diretório armazena os modelos treinados de Machine Learning do sistema de detecção DDoS.

## Estrutura

```
models/
├── hybrid/                        # Modelos híbridos básicos (legacy)
├── hybrid_advanced/               # Modelos híbridos avançados (ATUAL)
│   ├── hybrid_advanced_models.pkl
│   ├── performance_advanced.json
│   └── README.md
└── README.md                      # Este arquivo
```

## Sistema Atual

O sistema utiliza uma **arquitetura híbrida** com dois componentes:

### 🎯 DDoS Specialist (CIC-DDoS2019)
- **Função**: Detecção especializada de ataques DDoS
- **Performance**: F1-Score 99.99%
- **Modelos**: XGBoost + Random Forest

### 🛡️ General Detector (NF-UNSW-NB15-v3) 
- **Função**: Detecção generalista de intrusões de rede
- **Performance**: F1-Score 95.90%
- **Modelos**: XGBoost + Isolation Forest

## Treinamento

```bash
# Sistema completo (recomendado)
./make.ps1 train-hybrid-advanced

# Apenas processamento de dados
./make.ps1 setup-unsw

# Sistema básico (legacy)
./make.ps1 train-hybrid
```

## Performance

| Componente | Modelo | F1-Score | Precision | Recall |
|------------|--------|----------|-----------|---------|
| DDoS Specialist | XGBoost | 99.99% | 99.99% | 99.99% |
| DDoS Specialist | Random Forest | 99.96% | - | - |
| General Detector | XGBoost | 95.90% | 96.48% | 95.33% |
| General Detector | Isolation Forest | 66.20% | - | - |

**Nota**: Os arquivos .pkl não são incluídos no repositório devido ao tamanho. Execute o treinamento para gerar os modelos.
