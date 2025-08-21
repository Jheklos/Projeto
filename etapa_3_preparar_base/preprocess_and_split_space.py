import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

def preprocess_and_split(input_file):
    # Lê o CSV original com separador ;
    df = pd.read_csv(input_file, sep=";")

    # Remove a coluna 'file' (a primeira) se existir
    if df.columns[0].lower() == 'file':
        df = df.drop(columns=[df.columns[0]])

    # Verifica se agora a primeira coluna é a classe
    classes = df.iloc[:, 0].unique()
    if not set(classes).issubset({0, 1}):
        raise ValueError(f"❌ A primeira coluna deve conter apenas 0 ou 1 como classe. Encontrado: {classes}")

    # Separa a coluna de classe e as features
    target = df.iloc[:, 0]
    features = df.iloc[:, 1:]

    # Converte apenas as features para numérico
    features = features.apply(pd.to_numeric, errors="coerce")

    # Recombina e remove linhas com qualquer valor inválido
    df = pd.concat([target, features], axis=1).dropna()

    if df.shape[0] < 2:
        raise ValueError(f"❌ Dados insuficientes após limpeza: {df.shape[0]} linha(s).")

    print("✅ Dados prontos para divisão. Total de amostras:", df.shape[0])
    print("📊 Distribuição das classes:")
    print(df.iloc[:, 0].value_counts())

    # Divide em treino e teste com estratificação
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df.iloc[:, 0]
    )

    # Define nomes de saída
    base, _ = os.path.splitext(input_file)
    train_file = f"{base}_train.txt"
    test_file = f"{base}_test.txt"

    # Salva os arquivos com separador espaço, sem cabeçalho
    train_df.to_csv(train_file, sep=" ", index=False, header=False)
    test_df.to_csv(test_file, sep=" ", index=False, header=False)

    print(f"✅ Arquivo de treino salvo: {train_file}")
    print(f"✅ Arquivo de teste salvo: {test_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pré-processa CSV e divide em treino/teste no formato mELM")
    parser.add_argument("input_file", help="Caminho para o arquivo CSV de entrada (com ';')")
    args = parser.parse_args()

    preprocess_and_split(args.input_file)
