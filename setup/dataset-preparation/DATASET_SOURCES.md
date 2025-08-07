# Fontes de Conjuntos de Dados Académicos

Este documento especifica as fontes oficiais dos conjuntos de dados utilizados no projeto DDoS Mitigation Lab, incluindo informações académicas, publicações associadas e instruções de download manual.

## Conjuntos de Dados Utilizados

### 1. CIC-DDoS2019

**Fonte Académica**: Canadian Institute for Cybersecurity (CIC), University of New Brunswick
**Publicação**: Sharafaldin, Iman, Arash Habibi Lashkari, Saqib Hakak, and Ali A. Ghorbani. "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy." IEEE 53rd International Carnahan Conference on Security Technology, Chennai, India, 2019.
**DOI**: https://doi.org/10.1109/CCST.2019.8888419

**Descrição**: Conjunto de dados especializado em ataques DDoS, contendo tráfego de rede benigno e malicioso com 12 tipos diferentes de ataques DDoS incluindo DrDoS (DNS, LDAP, MSSQL, NetBIOS, NTP, SNMP, SSDP, UDP), SYN flood attacks, e outros.

**Características**:
- **Amostras totais**: 431,371 registos
- **Tráfego malicioso**: 333,538 amostras
- **Características**: 88 características de fluxo de rede
- **Formato**: CSV (13 ficheiros organizados em 2 directórios)
- **Directório 01-12**: DrDoS_SNMP, DrDoS_SSDP, DrDoS_UDP, Syn, TFTP, UDPLag
- **Directório 03-11**: LDAP, MSSQL, NetBIOS, Portmap, Syn, UDP, UDPLag

**Links de Download**:
- **Oficial**: https://www.unb.ca/cic/datasets/ddos-2019.html
- **Kaggle**: https://www.kaggle.com/datasets/aymenabb/ddos-evaluation-dataset-cic-ddos2019


### 2. NF-UNSW-NB15-v3 (NetFlow)

**Fonte Académica**: University of Queensland eSpace, Australian Centre for Cyber Security (ACCS), University of New South Wales (UNSW)
**Autores**: Luay, Majed; Layeghy, Siamak; Mohanad, Sarhan; Sayedehfaezeh, Hoseininoorbin; Moustafa, Nour; and Portmann, Marius
**DOI**: https://doi.org/10.48610/6e0eda1

**Descrição**: NF-UNSW-NB15-v3 é uma versão melhorada de conjuntos de dados baseados em NetFlow, incorporando 53 características extraídas que fornecem conhecimentos detalhados sobre fluxos de rede. O conjunto de dados inclui etiquetas binárias e multi-classe, distinguindo entre tráfego normal e nove tipos diferentes de ataques. Um dos aspectos-chave deste conjunto de dados é a inclusão de características temporais, que permitem uma análise mais detalhada do tráfego ao longo do tempo.

**Características**:
- **Características extraídas**: 53 características NetFlow detalhadas
- **Etiquetas**: Binárias (normal/ataque) e multi-classe (9 tipos de ataque)
- **Formato**: CSV estruturado, cada linha representa um fluxo de rede
- **Características temporais**: Timestamps precisos (início e fim de fluxo)
- **Estatísticas IAT**: Tempo entre chegadas de pacotes (mín, máx, média, desvio padrão)
- **Análise bidirecional**: Estatísticas origem-destino e destino-origem
- **Ataques incluídos**: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms

**Vantagens do formato NetFlow**:
- Dados agregados de fluxo em vez de pacotes individuais
- Formato padrão utilizado em infraestruturas de rede corporativas
- Menor overhead de processamento comparado com análise de pacotes
- Compatibilidade directa com ferramentas de monitorização de rede

**Links de Download**:
- **Oficial**: https://espace.library.uq.edu.au/view/UQ:6e0eda1
- **Kaggle**: https://www.kaggle.com/datasets/ndayisabae/nf-unsw-nb15-v3

## Estrutura de Directórios Local

Após o download manual, os conjuntos de dados devem ser organizados da seguinte forma:

```
setup/dataset-preparation/
├── CIC-DDoS2019/                    # Conjunto de dados CIC-DDoS2019
│   ├── 01-12/                       # Directório de dados 01-12
│   │   ├── DrDoS_SNMP.csv          # Ataques SNMP DRDoS
│   │   ├── DrDoS_SSDP.csv          # Ataques SSDP DRDoS
│   │   ├── DrDoS_UDP.csv           # Ataques UDP DRDoS
│   │   ├── Syn.csv                 # Ataques SYN Flood
│   │   ├── TFTP.csv                # Ataques TFTP
│   │   └── UDPLag.csv              # Ataques UDP Lag
│   └── 03-11/                       # Directório de dados 03-11
│       ├── LDAP.csv                # Ataques LDAP
│       ├── MSSQL.csv               # Ataques MSSQL
│       ├── NetBIOS.csv             # Ataques NetBIOS
│       ├── Portmap.csv             # Ataques Portmap
│       ├── Syn.csv                 # Ataques SYN Flood
│       ├── UDP.csv                 # Ataques UDP
│       └── UDPLag.csv              # Ataques UDP Lag
├── NF-UNSW-NB15-v3/                 # Conjunto de dados NF-UNSW-NB15-v3
│   ├── NetFlow_v3_Features.csv     # Ficheiro de características NetFlow v3
│   └── NF-UNSW-NB15-v3.csv         # Ficheiro principal do dataset
└── README.md                        # Este ficheiro
```

## Instruções de Download Manual

### Para CIC-DDoS2019:

1. **Aceder ao site oficial**: https://www.unb.ca/cic/datasets/ddos-2019.html
2. **Preencher formulário de registo** (nome, email, instituição)
3. **Descarregar o ficheiro ZIP**: `DDoS2019.zip` (~2.3 GB)
4. **Extrair para**: `setup/dataset-preparation/CIC-DDoS2019/`
5. **Verificar estrutura**: Deve conter directórios `01-12/` e `03-11/` com 13 ficheiros CSV total

**Alternativa via Kaggle**:
1. Criar conta no Kaggle
2. Aceder: https://www.kaggle.com/datasets/aymenabb/ddos-evaluation-dataset-cic-ddos2019
3. Descarregar dataset
4. Extrair para localização correcta

### Para NF-UNSW-NB15-v3:

1. **Aceder ao repositório oficial**: https://espace.library.uq.edu.au/view/UQ:6e0eda1
2. **Descarregar ficheiro**: `NF-UNSW-NB15-v3.zip` (~800 MB)
3. **Extrair para**: `setup/dataset-preparation/NF-UNSW-NB15-v3/`
4. **Verificar estrutura**: Deve conter 2 ficheiros CSV (NetFlow_v3_Features.csv + NF-UNSW-NB15-v3.csv)

**Alternativa via Kaggle**:
1. Criar conta no Kaggle
2. Aceder: https://www.kaggle.com/datasets/ndayisabae/nf-unsw-nb15-v3
3. Descarregar dataset
4. Extrair para localização correcta

## Verificação da Instalação

Após o download e extracção, verifique a instalação executando:

```bash
cd setup/dataset-preparation
python prepare_datasets.py --check
```

O sistema deve detectar:
- **CIC-DDoS2019**: 13 ficheiros CSV (6 ficheiros em 01-12/ + 7 ficheiros em 03-11/)
- **NF-UNSW-NB15-v3**: 2 ficheiros CSV (NetFlow_v3_Features.csv + NF-UNSW-NB15-v3.csv)

## Citações Académicas

Se utilizar estes conjuntos de dados em investigação académica, por favor cite as publicações originais:

**Para CIC-DDoS2019**:
```bibtex
@inproceedings{sharafaldin2019developing,
  title={Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy},
  author={Sharafaldin, Iman and Lashkari, Arash Habibi and Hakak, Saqib and Ghorbani, Ali A},
  booktitle={IEEE 53rd International Carnahan Conference on Security Technology},
  pages={1--8},
  year={2019},
  address={Chennai, India},
  organization={IEEE},
  doi={10.1109/CCST.2019.8888419}
}
```

**Para NF-UNSW-NB15-v3 (versão NetFlow)**:
```bibtex
@dataset{luay2023nfunsw,
  title={NF-UNSW-NB15-v3},
  author={Luay, Majed and Layeghy, Siamak and Sarhan, Mohanad and Hoseininoorbin, Sayedehfaezeh and Moustafa, Nour and Portmann, Marius},
  year={2023},
  publisher={University of Queensland},
  doi={10.48610/6e0eda1},
  url={https://espace.library.uq.edu.au/view/UQ:6e0eda1},
  note={Enhanced NetFlow-based dataset with 53 extracted features for network intrusion detection}
}
```

**Para o dataset original UNSW-NB15 (se referenciado)**:
```bibtex
@inproceedings{moustafa2015unsw,
  title={UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)},
  author={Moustafa, Nour and Slay, Jill},
  booktitle={2015 Military Communications and Information Systems Conference (MilCIS)},
  pages={1--6},
  year={2015},
  organization={IEEE}
}
```

## Licenças e Termos de Uso

- **CIC-DDoS2019**: 
  - **Licença**: Redistribuição livre permitida (redistribuir, republicar, espelhar)
  - **Requisito obrigatório**: Citação do dataset CIC-DDoS2019 e artigo relacionado
  - **Uso**: Permitido para qualquer fim com citação adequada
  - **Redistribuição**: Qualquer uso ou redistribuição deve incluir citação
- **NF-UNSW-NB15-v3**: 
  - **Acesso**: Acesso Aberto (Open Access)
  - **Licença**: Reutilização Permitida com Restrição de Uso Comercial
  - **Requisitos**: Citação obrigatória em qualquer reutilização
  - **Restrição**: Apenas para fins não comerciais
  - **Reconhecimento**: Citação completa conforme apresentada no registo UQ eSpace
