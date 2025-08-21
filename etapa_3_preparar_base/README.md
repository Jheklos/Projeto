# Preparar base de dados para rodar os algorítmos


converter para rodar a rede neural

adicionar bibliotecas:

python3 -m venv myenv   

source myenv/bin/activate 

pip install scikit-learn    

Depois de colocar os beniguinis e malwere em um só csv

Execultar arquivo:

coloque o CSV na mesma pasta do script e rode:

python3 preprocess_and_split_space.py AQUI_o_ARQUIVO.csv


-------------------------------------------------------------------

converter para rodar o smv

execulte o comando ( o CSV na mesma pasta do script)

python converter_libsvm.py AQUI_o_ARQUIVO.csv NOME_CSV.libsvm

ou com o caminho ( o CSV em outra pasta)

python converter_libsvm.py /caminho/para/AQUI_o_ARQUIVO.csv /caminho/para/NOME_CSV.libsvm
