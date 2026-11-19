"""Camada de dados: limpeza do CSV cru e leitura do CSV limpo."""
from pathlib import Path

import pandas as pd
import streamlit as st

ARQUIVO_CRU = "sales.csv"
ARQUIVO_LIMPO = "sales_clean.csv"
COLUNAS_TEXTO = ["cliente", "cidade", "categoria"]


def preparar_dados(input_path: str = ARQUIVO_CRU, output_path: str = ARQUIVO_LIMPO) -> pd.DataFrame:
    """Lê o CSV cru, padroniza e salva o CSV limpo (sep ';' e decimal ',')."""
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Arquivo fonte não encontrado: {input_path}")

    df = pd.read_csv(input_path)
    df["data"] = pd.to_datetime(df["data"])
    df["status"] = df["status"].str.lower().str.strip()
    for col in COLUNAS_TEXTO:
        if col in df.columns:
            df[col] = df[col].str.strip()
    df = df.drop_duplicates()

    # Premissa técnica: receita = quantidade * preço unitário
    if "receita" not in df.columns:
        df["receita"] = df["quantidade"] * df["preco_unitario"]

    df.to_csv(output_path, sep=";", decimal=",", index=False)
    return df


@st.cache_data
def _ler_csv_limpo(path: str, mtime: float) -> pd.DataFrame:
    # mtime entra na chave do cache (sem '_' para o Streamlit considerá-lo): CSV regerado invalida o cache
    return pd.read_csv(path, sep=";", decimal=",", parse_dates=["data"])


def carregar_dados(path: str = ARQUIVO_LIMPO, origem: str = ARQUIVO_CRU) -> pd.DataFrame:
    """Lê o CSV limpo; se ainda não existir e houver o cru, gera antes."""
    arquivo = Path(path)
    if not arquivo.exists():
        if not Path(origem).exists():
            return pd.DataFrame()
        preparar_dados(origem, path)
    return _ler_csv_limpo(path, arquivo.stat().st_mtime)


if __name__ == "__main__":
    limpo = preparar_dados()
    print(f"{ARQUIVO_LIMPO} gerado: {len(limpo)} registros")
