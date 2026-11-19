"""Camada de métricas: uma função por pergunta de negócio.

Regra: o Python calcula, a IA só explica. Painel e chat usam estas mesmas funções,
por isso os números batem. Filtros comuns: categoria e período (data_inicio/data_fim, AAAA-MM-DD, inclusivos).
"""
import unicodedata

import pandas as pd

STATUS_CONCLUIDO = "concluído"
STATUS_CANCELADO = "cancelado"
DIMENSOES = ("categoria", "cidade", "mes")


def _normalizar(texto: str) -> str:
    """Compara categorias sem acento, maiúsculas nem plural simples ("esporte" == "Esportes")."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower().strip()
    return t[:-1] if t.endswith("s") else t


def aplicar_filtros(df: pd.DataFrame, categoria: str | None = None,
                    data_inicio: str | None = None, data_fim: str | None = None) -> pd.DataFrame:
    """Filtra por categoria (sem diferenciar maiúsculas, acentos e plural simples) e por período."""
    mask = pd.Series(True, index=df.index)
    if categoria:
        mask &= df["categoria"].map(_normalizar) == _normalizar(categoria)
    if data_inicio:
        mask &= df["data"] >= pd.Timestamp(data_inicio)
    if data_fim:
        # Inclusivo: considera o dia inteiro de data_fim
        mask &= df["data"] < pd.Timestamp(data_fim) + pd.Timedelta(days=1)
    return df[mask]


def filtrar_concluidos(df: pd.DataFrame, categoria: str | None = None,
                       data_inicio: str | None = None, data_fim: str | None = None) -> pd.DataFrame:
    df_calc = aplicar_filtros(df, categoria, data_inicio, data_fim)
    return df_calc[df_calc["status"] == STATUS_CONCLUIDO]


def obter_receita_total(df: pd.DataFrame, categoria: str | None = None,
                        data_inicio: str | None = None, data_fim: str | None = None) -> float:
    return round(float(filtrar_concluidos(df, categoria, data_inicio, data_fim)["receita"].sum()), 2)


def obter_total_pedidos(df: pd.DataFrame, categoria: str | None = None,
                        data_inicio: str | None = None, data_fim: str | None = None) -> int:
    return int(filtrar_concluidos(df, categoria, data_inicio, data_fim).shape[0])


def obter_ticket_medio(df: pd.DataFrame, categoria: str | None = None,
                       data_inicio: str | None = None, data_fim: str | None = None) -> float:
    df_calc = filtrar_concluidos(df, categoria, data_inicio, data_fim)
    if df_calc.empty:
        return 0.0
    return round(float(df_calc["receita"].sum() / len(df_calc)), 2)


def obter_clientes_unicos(df: pd.DataFrame, categoria: str | None = None,
                          data_inicio: str | None = None, data_fim: str | None = None) -> int:
    return int(filtrar_concluidos(df, categoria, data_inicio, data_fim)["cliente"].nunique())


def obter_receita_media_por_cliente(df: pd.DataFrame, categoria: str | None = None,
                                    data_inicio: str | None = None, data_fim: str | None = None) -> float:
    """Evolução: receita dos pedidos concluídos / clientes únicos."""
    df_calc = filtrar_concluidos(df, categoria, data_inicio, data_fim)
    clientes = df_calc["cliente"].nunique()
    if clientes == 0:
        return 0.0
    return round(float(df_calc["receita"].sum() / clientes), 2)


def obter_taxa_cancelamento(df: pd.DataFrame, categoria: str | None = None,
                            data_inicio: str | None = None, data_fim: str | None = None) -> float:
    """Cancelados / total de pedidos (todos os status), em %. Única métrica fora da regra 'só concluídos'."""
    df_calc = aplicar_filtros(df, categoria, data_inicio, data_fim)
    if df_calc.empty:
        return 0.0
    return round(float((df_calc["status"] == STATUS_CANCELADO).mean() * 100), 2)


def obter_top_cidades(df: pd.DataFrame, limite: int = 5, categoria: str | None = None,
                      data_inicio: str | None = None, data_fim: str | None = None) -> dict:
    """Evolução: cidades com maior receita (pedidos concluídos)."""
    serie = (filtrar_concluidos(df, categoria, data_inicio, data_fim)
             .groupby("cidade")["receita"].sum().nlargest(int(limite)))
    return serie.round(2).to_dict()


def obter_receita_mensal(df: pd.DataFrame, categoria: str | None = None,
                         data_inicio: str | None = None, data_fim: str | None = None) -> dict:
    df_calc = filtrar_concluidos(df, categoria, data_inicio, data_fim)
    serie = df_calc.groupby(df_calc["data"].dt.to_period("M"))["receita"].sum().sort_index()
    return {str(p): round(float(v), 2) for p, v in serie.items()}


def obter_resumo_por(df: pd.DataFrame, dimensao: str = "categoria", categoria: str | None = None,
                     data_inicio: str | None = None, data_fim: str | None = None) -> dict:
    """Todas as métricas por categoria, cidade ou mês em uma chamada (evita N rodadas do LLM)."""
    if dimensao not in DIMENSOES:
        raise ValueError(f"dimensao deve ser uma de {DIMENSOES}")
    base = aplicar_filtros(df, categoria, data_inicio, data_fim)

    def chave(d: pd.DataFrame) -> pd.Series:
        return d["data"].dt.to_period("M").astype(str) if dimensao == "mes" else d[dimensao]

    concl = base[base["status"] == STATUS_CONCLUIDO]
    res = concl.groupby(chave(concl)).agg(
        receita=("receita", "sum"), pedidos=("receita", "size"), clientes_unicos=("cliente", "nunique"))
    res["ticket_medio"] = res["receita"] / res["pedidos"]
    res["receita_media_por_cliente"] = res["receita"] / res["clientes_unicos"]
    taxa = (base["status"] == STATUS_CANCELADO).groupby(chave(base)).mean().mul(100)
    res = res.join(taxa.rename("taxa_cancelamento_pct"), how="outer").fillna(0)
    res = res.sort_index() if dimensao == "mes" else res.sort_values("receita", ascending=False)
    inteiros = {"pedidos", "clientes_unicos"}
    return {str(k): {c: int(v) if c in inteiros else round(float(v), 2) for c, v in linha.items()}
            for k, linha in res.iterrows()}
