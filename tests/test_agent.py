import inspect
import json

import pytest
from openai.types.chat import ChatCompletion

import agent
import metrics


def _resposta(conteudo=None, tool_calls=None) -> ChatCompletion:
    msg = {"role": "assistant", "content": conteudo}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(tool_calls)
        ]
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
        "choices": [{"index": 0, "finish_reason": "stop", "message": msg}],
    })


def test_toda_tool_tem_funcao_e_parametros_validos():
    for tool in agent.TOOLS:
        fn = tool["function"]
        func = getattr(metrics, fn["name"])
        assinatura = set(inspect.signature(func).parameters) - {"df"}
        assert set(fn["parameters"]["properties"]) <= assinatura, fn["name"]
        assert fn["description"]


def test_executar_tool_por_nome(df):
    assert agent._executar_tool(df, "obter_receita_total", '{"categoria": "Cat1"}') == "300.0"
    assert agent._executar_tool(df, "obter_total_pedidos", "") == "3"  # argumentos vazios
    assert agent._executar_tool(df, "obter_total_pedidos", '{"categoria": null}') == "3"
    assert json.loads(agent._executar_tool(df, "obter_top_cidades", '{"limite": 1}')) == {"Cidade1": 400.0}


def test_executar_tool_bloqueia_funcao_nao_declarada(df):
    assert "não disponível" in agent._executar_tool(df, "filtrar_concluidos", "{}")
    assert "Erro ao executar" in agent._executar_tool(df, "obter_resumo_por", '{"dimensao": "estado"}')


def test_fluxo_tool_calling_numero_vem_do_python(df, monkeypatch):
    roteiro = iter([
        _resposta(tool_calls=[("obter_resumo_por", {"dimensao": "categoria"})]),
        _resposta(conteudo="Cat1: R$ 300,00 · Cat2: R$ 300,00"),
    ])
    enviados = []

    def fake(p, mensagens, tool_choice):
        enviados.append([dict(m) if isinstance(m, dict) else m for m in mensagens])
        return next(roteiro)

    monkeypatch.setattr(agent, "_chamar_llm", fake)
    passos = []
    resposta = agent.executar_agente(df, [{"role": "user", "content": "receita por categoria"}], on_step=passos.append)

    assert resposta.startswith("Cat1")
    resultado_tool = json.loads(enviados[1][-1]["content"])
    assert resultado_tool["Cat1"]["receita"] == 300.0
    assert any("obter_resumo_por" in p for p in passos)


def test_ultima_rodada_forca_resposta(df, monkeypatch):
    escolhas = []

    def fake(p, mensagens, tool_choice):
        escolhas.append(tool_choice)
        if tool_choice == "none":
            return _resposta(conteudo="final")
        return _resposta(tool_calls=[("obter_receita_total", {})])

    monkeypatch.setattr(agent, "_chamar_llm", fake)
    assert agent.executar_agente(df, [{"role": "user", "content": "x"}], max_iter=3) == "final"
    assert escolhas == ["auto", "auto", "none"]


@pytest.mark.parametrize("env, esperado", [
    ({"OPENROUTER_API_KEY": "sk-or-x"}, "openrouter"),
    ({"GEMINI_API_KEY": "abc"}, "gemini"),
    ({"OPENROUTER_API_KEY": "sua_chave_aqui", "GEMINI_API_KEY": "abc"}, "gemini"),
    ({"LLM_PROVIDER": "gemini", "OPENROUTER_API_KEY": "sk-or-x"}, "gemini"),
    ({}, "openrouter"),
])
def test_provedor_ativo(monkeypatch, env, esperado):
    for var in ("LLM_PROVIDER", "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert agent.provedor_ativo().id == esperado


def test_headers_gemini_por_modo(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.abc")
    base = dict(id="gemini", nome="G", base_url="u", var_chave="GEMINI_API_KEY", modelo="m")
    p = agent.Provedor(**base, auth="auto", url_nativa="n")
    h = p.headers_requisicao("nativo")
    assert h["x-goog-api-key"] == "AQ.abc" and type(h["Authorization"]).__name__ == "Omit"
    assert p.headers_requisicao("bearer") == {}
    assert p.modos_auth() == ["bearer", "nativo"]
    assert agent.Provedor(**base, auth="nativo").modos_auth() == ["nativo"]
    assert agent.Provedor(**base, auth="header").modos_auth() == ["nativo"]  # nome antigo
    assert agent.Provedor(**base, auth="bearer").modos_auth() == ["bearer"]
    assert agent.Provedor(**base, auth="auto").modos_auth() == ["bearer"]  # sem API nativa (OpenRouter)


@pytest.mark.parametrize("modelo, suporta", [
    ("gemini-1.5-flash", False), ("gemini-2.0-flash", False), ("gemini-2.5-flash", True),
    ("gemini-3.8-flash", True), ("openrouter/free", True),
])
def test_reasoning_so_para_modelos_com_raciocinio(modelo, suporta):
    assert agent._suporta_raciocinio(modelo) is suporta


def test_categoria_todas_vira_total_geral(df):
    assert agent._executar_tool(df, "obter_receita_total", '{"categoria": "Todas"}') == "600.0"
