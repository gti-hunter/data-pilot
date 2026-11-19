import pandas as pd

from data import carregar_dados, preparar_dados

CSV_CRU = (
    "id_pedido,data,cliente,cidade,categoria,quantidade,preco_unitario,status\n"
    "P1,2026-01-10,A,São Paulo,Móveis,2,10.5, Concluído \n"
    "P1,2026-01-10,A,São Paulo,Móveis,2,10.5, Concluído \n"
    "P2,2026-01-11,B, Curitiba ,Vestuário,1,3.25,CANCELADO\n"
)


def test_preparar_e_carregar(tmp_path):
    cru, limpo = tmp_path / "sales.csv", tmp_path / "sales_clean.csv"
    cru.write_text(CSV_CRU, encoding="utf-8")

    df = carregar_dados(str(limpo), str(cru))  # gera o limpo automaticamente

    assert len(df) == 2  # duplicado removido
    assert df["status"].tolist() == ["concluído", "cancelado"]
    assert df["cidade"].tolist() == ["São Paulo", "Curitiba"]
    assert df["receita"].tolist() == [21.0, 3.25]
    assert pd.api.types.is_datetime64_any_dtype(df["data"])
    # Formato do arquivo limpo: ';' como separador e ',' como decimal
    assert limpo.read_text(encoding="utf-8").splitlines()[1].endswith(";concluído;21,0")


def test_carregar_sem_arquivos_retorna_vazio(tmp_path):
    assert carregar_dados(str(tmp_path / "x.csv"), str(tmp_path / "y.csv")).empty


def test_preparar_retorna_dataframe(tmp_path):
    cru = tmp_path / "sales.csv"
    cru.write_text(CSV_CRU, encoding="utf-8")
    assert len(preparar_dados(str(cru), str(tmp_path / "out.csv"))) == 2
