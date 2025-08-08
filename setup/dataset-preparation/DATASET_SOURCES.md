# Datasets Académicos do Projeto DDoS Mitigation Lab

Este documento identifica os datasets utilizados, como descarregar manualmente, estrutura local recomendada, características principais e citações académicas.

---

## CIC-DDoS2019
- **Fonte**: Canadian Institute for Cybersecurity (CIC), University of New Brunswick
- **Dataset**: [Site oficial](https://www.unb.ca/cic/datasets/ddos-2019.html)
- **Download**: [Kaggle - cicddos2019.parquet (limpo e tratado)](https://www.kaggle.com/datasets/dhoogla/cicddos2019/data)
- **Como obter**: Descarregar e extrair o 'cicddos2019.parquet' para `setup/dataset-preparation/CIC-DDoS2019/`
- **Estrutura**:
  ```
  setup/dataset-preparation/
  └── CIC-DDoS2019/
      └── cicddos2019.parquet
  ```
- **Características**: 138.007 registos, 67 colunas, formato Parquet
- **Citação**:
  ```bibtex
  @inproceedings{sharafaldin2019developing,
    title={Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy},
    author={Sharafaldin, Iman and Lashkari, Arash Habibi and Hakak, Saqib and Ghorbani, Ali A},
    booktitle={IEEE 53rd International Carnahan Conference on Security Technology},
    pages={1--8},
    year={2019},
    doi={10.1109/CCST.2019.8888419}
  }
  ```

---

## CIC-BoT-IoT
- **Fonte**: UNSW-BoT-IoT, with CICFlowmeter features, by the University of Queensland
- **Autores**: Dr. Mohanad Sarhan, Dr. Siamak Layeghy & Dr. Marius Portmann
- **Preprint**: [arXiv:2104.07183](https://arxiv.org/abs/2104.07183)
- **Download**: [Kaggle - CIC-BoT-IoT (limpo e tratado)](https://www.kaggle.com/datasets/dhoogla/cicbotiot)
- **Como obter**: Descarregar e extrair para `setup/dataset-preparation/CIC-BoT-IoT/`
- **Estrutura**:
  ```
  setup/dataset-preparation/
  └── CIC-BoT-IoT/
      ├── CIC-BoT-IoT-V2.parquet
      └── CICFLowMeter Features.csv
  ```
- **Características**: 11.503.556 registos, 79 colunas, formato Parquet + CSV
- **Citação**:
  ```bibtex
  @article{sarhan2021cicbotiot,
    title={CIC-BoT-IoT: Botnet Dataset with Flow Meter Features},
    author={Sarhan, Mohanad and Layeghy, Siamak and Portmann, Marius},
    journal={arXiv preprint arXiv:2104.07183},
    year={2021}
  }
  ```

---

## NF-UNSW-NB15-v3 (NetFlow)
- **Fonte**: [UQ eSpace](https://espace.library.uq.edu.au/view/UQ:6e0eda1) | [Kaggle](https://www.kaggle.com/datasets/ndayisabae/nf-unsw-nb15-v3)
- **Autores**: Luay, Majed; Layeghy, Siamak; Sarhan, Mohanad; Hoseininoorbin, Sayedehfaezeh; Moustafa, Nour; Portmann, Marius
- **Como obter**: Descarregar e extrair para `setup/dataset-preparation/NF-UNSW-NB15-v3/`
- **Estrutura**:
  ```
  setup/dataset-preparation/
  └── NF-UNSW-NB15-v3/
      ├── NetFlow_v3_Features.csv
      └── NF-UNSW-NB15-v3.csv
  ```
- **Características**: 2.365.424 registos, 55 colunas, formato CSV
- **Citação**:
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
