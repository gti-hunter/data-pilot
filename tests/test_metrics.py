import pytest

import metrics as m


def test_metricas_consideram_apenas_concluidos(df):
    assert m.obter_receita_total(df) == 600.0
    assert m.obter_total_pedidos(df) == 3
    assert m.obter_ticket_medio(df) == 200.0
    assert m.obter_clientes_unicos(df) == 2
    assert m.obter_receita_media_por_cliente(df) == 300.0


def test_taxa_cancelamento_sobre_todos_os_pedidos(df):
    assert m.obter_taxa_cancelamento(df) == 20.0


def test_filtro_categoria_ignora_maiusculas(df):
    assert m.obter_receita_total(df, categoria="cat1") == 300.0
    assert m.obter_total_pedidos(df, categoria="CAT1") == 2


def test_filtro_periodo_inclusivo(df):
    assert m.obter_receita_total(df, data_inicio="2026-02-01", data_fim="2026-02-28") == 300.0
    assert m.obter_taxa_cancelamento(df, data_inicio="2026-02-01", data_fim="2026-02-28") == 50.0
    assert m.obter_receita_total(df, data_fim="2026-01-20") == 300.0  # inclui o próprio dia


def test_filtro_sem_dados_retorna_zero(df):
    filtros = {"categoria": "Inexistente"}
    assert m.obter_receita_total(df, **filtros) == 0.0
    assert m.obter_ticket_medio(df, **filtros) == 0.0
    assert m.obter_receita_media_por_cliente(df, **filtros) == 0.0
    assert m.obter_taxa_cancelamento(df, **filtros) == 0.0
    assert m.obter_receita_mensal(df, **filtros) == {}


def test_top_cidades_e_receita_mensal(df):
    assert m.obter_top_cidades(df, limite=2) == {"Cidade1": 400.0, "Cidade2": 200.0}
    assert m.obter_receita_mensal(df) == {"2026-01": 300.0, "2026-02": 300.0}


def test_resumo_por_categoria_bate_com_metricas_individuais(df):
    resumo = m.obter_resumo_por(df, "categoria")
    for cat, linha in resumo.items():
        assert linha["receita"] == m.obter_receita_total(df, categoria=cat)
        assert linha["pedidos"] == m.obter_total_pedidos(df, categoria=cat)
        assert linha["taxa_cancelamento_pct"] == m.obter_taxa_cancelamento(df, categoria=cat)
    assert sum(linha["receita"] for linha in resumo.values()) == m.obter_receita_total(df)


def test_resumo_por_dimensao_invalida(df):
    with pytest.raises(ValueError):
        m.obter_resumo_por(df, "estado")


def test_categoria_tolera_acento_e_plural(df):
    d = df.assign(categoria=["Eletrônicos", "Eletrônicos", "Esportes", "Esportes", "Esportes"])
    assert m.obter_receita_total(d, categoria="eletronico") == 300.0
    assert m.obter_receita_total(d, categoria="esporte") == 300.0
    assert m.obter_receita_total(d, categoria="ESPORTES") == 300.0
