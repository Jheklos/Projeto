# Montar Planilhas

você precisa criar essa estrutura de pastas:

etapa_2_analise_estatica/
├── montar_planilhas.py
├── PegaDlls.py
├── PegaIAT.py
├── PegaResource.py
../
└── etapa_1_virustotal_api/
    └── APT/
        ├── benign/
        │   └── analysis/
        │       ├── benign1.txt
        │       └── benign2.txt
        └── malware/
            └── analysis/
                ├── malware1.txt
                └── malware2.txt



Para criar a estrutura, dentro da pasta etapa_2_analise_estatica rode o comando:

mkdir -p ../etapa_1_virustotal_api/APT/benign/analysis
mkdir -p ../etapa_1_virustotal_api/APT/malware/analysis



Coloque os arquivos .txt do pescanner dentro de:

../etapa_1_virustotal_api/APT/benign/analysis/

../etapa_1_virustotal_api/APT/malware/analysis/

Dentro da pasta etapa_2_analise_estatica Rode o script principal:

python3 montar_planilhas.py


