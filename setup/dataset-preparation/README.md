# Sistema de Preparação de Conjuntos de Dados

Sistema completo de preparação de conjuntos de dados para o DDoS Mitigation Lab. Este sistema inclui tanto os conjuntos de dados brutos como as ferramentas de processamento numa localização organizada.


## Visão Geral do Sistema

Este diretório contém:
- **Conjuntos de dados brutos**: NF-UNSW-NB15-v3 e CIC-DDoS2019 
- **Scripts de processamento**: Processadores profissionais de conjuntos de dados
- **Ferramentas de integração**: Geradores de configuração multi-conjunto de dados
- **Documentação**: Guias completos de configuração e utilização

## Fontes e Instruções de Download dos Conjuntos de Dados

> **Nota:** As instruções detalhadas de download manual, fontes académicas, citações e estrutura de diretórios dos datasets estão documentadas em [`DATASET_SOURCES.md`](./DATASET_SOURCES.md).

Antes de executar qualquer processamento, é obrigatório efetuar o download manual dos datasets originais e colocá-los nas localizações corretas, conforme descrito nesse documento.

**Resumo do procedimento:**

1. Consulte o ficheiro [`DATASET_SOURCES.md`](./DATASET_SOURCES.md) para obter os links oficiais, instruções de registo/download e estrutura de pastas esperada.
2. Faça o download manual dos ficheiros ZIP dos datasets (CIC-DDoS2019 e NF-UNSW-NB15-v3) a partir das fontes oficiais ou alternativas (Kaggle).
3. Extraia os ficheiros para as pastas indicadas:
   - `setup/dataset-preparation/CIC-DDoS2019/`
   - `setup/dataset-preparation/NF-UNSW-NB15-v3/`
4. Verifique a estrutura de diretórios e a presença dos ficheiros CSV conforme indicado.
5. Para confirmar que os datasets estão corretamente instalados, execute:

   ```bash
   python3 prepare_datasets.py --check
   ```

O comando acima **apenas verifica se os datasets estão presentes e prontos**. Para processar os dados, basta correr o script sem argumentos.

   ```bash
   python3 prepare_datasets.py
   ```

## Estrutura do Diretório

```
setup/dataset-preparation/
├── NF-UNSW-NB15-v3/            # Ficheiros brutos do conjunto de dados NF-UNSW NetFlow
├── CIC-DDoS2019/               # Ficheiros brutos do conjunto de dados CIC-DDoS2019
├── process_nf_unsw.py          # Processador NF-UNSW
├── process_cic_ddos.py         # Processador CIC-DDoS
├── integrate_datasets.py       # Integração de conjuntos de dados
├── prepare_datasets.py         # Pipeline principal
├── DATASET_SOURCES.md          # Fontes de conjuntos de dados e informação de descarregamento
└── README.md                   # Este ficheiro

Diretórios de saída (criados durante o processamento):
src/datasets/processed/
├── X_nf_unsw.npy              # Características NF-UNSW
├── y_nf_unsw.npy              # Etiquetas NF-UNSW
├── X_cic_ddos.npy             # Características CIC-DDoS
├── y_cic_ddos.npy             # Etiquetas CIC-DDoS
├── *_metadata.json            # Metadados dos conjuntos de dados
└── integrated/                # Configurações combinadas
```

## Configurações de Conjuntos de Dados

O sistema cria quatro configurações distintas de treino:

### 1. Configuração Combinada
- Todos os conjuntos de dados unidos num único conjunto de treino
- Adequada para treino de modelo unificado
- Representação equilibrada de diferentes tipos de ataque

### 2. Configuração Ensemble Separada
- Mantém fronteiras de conjuntos de dados para aprendizagem ensemble
- Permite treino de modelos especializados em diferentes conjuntos de dados
- Suporta abordagens de meta-aprendizagem

### 3. Configurações Individuais
- Cada conjunto de dados disponível separadamente
- Permite treino de modelos especializados
- Útil para análise comparativa

### 4. Versões Dimensionadas
- Todas as configurações incluem versões padronizadas
- Dimensionamento de características aplicado usando StandardScaler
- Dimensionadores guardados para inferência consistente

## Garantia de Qualidade de Dados

O sistema implementa medidas abrangentes de qualidade de dados:

- **Tratamento de Valores Ausentes**: Imputação da mediana para características numéricas
- **Processamento de Valores Infinitos**: Substituição por NaN seguida de imputação
- **Variância de Características**: Remoção de características com variância zero
- **Gestão de Memória**: Processamento em lotes para grandes conjuntos de dados
- **Validação de Dados**: Verificação da integridade dos ficheiros de saída

## Especificações Técnicas

- **Eficiência de Memória**: Processamento em lotes para conjuntos de dados que excedem a RAM disponível
- **Alinhamento de Características**: Tratamento automático de diferentes dimensões de características
- **Tratamento de Erros Robusto**: Registo e recuperação abrangente de erros
- **Arquitetura Escalável**: Design modular para extensão fácil

## Metadados de Saída

Cada conjunto de dados processado inclui metadados abrangentes:

```json
{
  "dataset": "nome_conjunto_dados",
  "processing_date": "timestamp_ISO",
  "total_samples": 123456,
  "feature_count": 42,
  "attack_ratio": 0.8534,
  "benign_samples": 18000,
  "attack_samples": 105456,
  "feature_names": ["caracteristica1", "caracteristica2", "..."]
}
```

## Requisitos

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Espaço em disco suficiente para conjuntos de dados processados

## Resolução de Problemas

**Problemas de Memória**
- Reduzir o parâmetro batch_size no processador CIC-DDoS
- Garantir espaço em disco suficiente para ficheiros temporários

**Conjuntos de Dados Ausentes**
- Verificar colocação de conjuntos de dados nos diretórios corretos
- Verificar permissões e formatos de ficheiros

**Erros de Processamento**
- Rever registos na saída do terminal
- Verificar processing_report.json para informação detalhada de erros

## Extensão

O design modular permite adição fácil de novos conjuntos de dados:

1. Criar um novo script de processamento seguindo o padrão existente
2. Adicionar lógica de detecção de conjunto de dados ao pipeline principal
3. Atualizar módulo de integração para tratar novo conjunto de dados

Este sistema fornece uma base robusta para preparação e análise abrangente de conjuntos de dados de segurança de rede.
