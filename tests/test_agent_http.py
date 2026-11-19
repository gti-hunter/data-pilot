"""Integração do agente contra um servidor HTTP local que imita o Gemini:
API nativa (generateContent + x-goog-api-key) e endpoint compatível com OpenAI (Bearer)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openai import OpenAI

import agent

MODELOS = ["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash",
           "gemini-3-flash-preview", "gemini-flash-latest", "gemini-3.1-flash-tts-preview"]
NATIVO, COMPAT = "generateContent", "chat/completions"


def _erro(status: int, mensagem: str, codigo: str = "INVALID_ARGUMENT", lista: bool = False):
    corpo = {"error": {"code": status, "message": mensagem, "status": codigo}}
    return status, [corpo] if lista else corpo  # o endpoint compatível devolve o erro dentro de uma lista


def _nativo(*partes):
    return 200, {"candidates": [{"content": {"role": "model", "parts": list(partes)}}], "modelVersion": "gemini-x"}


def _compat(conteudo):
    return 200, {"id": "x", "object": "chat.completion", "created": 0, "model": "gemini-x",
                 "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": conteudo}}]}


class _Servidor:
    def __init__(self, respostas):
        self.respostas, self.requisicoes = list(respostas), []
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _enviar(self, status, payload):
                dados = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(dados)))
                self.end_headers()
                self.wfile.write(dados)

            def do_GET(self):  # lista de modelos: formato OpenAI (compatível) ou nativo
                if "/openai/" in self.path:
                    self._enviar(200, {"object": "list", "data": [
                        {"id": f"models/{m}", "object": "model", "created": 0, "owned_by": "google"} for m in MODELOS]})
                else:
                    metodos = ["generateContent"]
                    self._enviar(200, {"models": [{"name": f"models/{m}", "supportedGenerationMethods": metodos}
                                                  for m in MODELOS]})

            def do_POST(self):
                corpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                srv.requisicoes.append({"path": self.path, "corpo": corpo,
                                        "headers": {k.lower(): v for k, v in self.headers.items()}})
                resposta = srv.respostas.pop(0)
                if len(resposta) == 3:  # (status, payload, atraso_s)
                    time.sleep(resposta[2])
                self._enviar(*resposta[:2])

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"


@pytest.fixture
def gemini(monkeypatch):
    agent._AJUSTES.clear()

    for var in ("GEMINI_FALLBACK_MODELS", "OPENROUTER_FALLBACK_MODELS", "OPENROUTER_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)

    def montar(respostas, modelo="gemini-3.8-flash", chave="AQ.chave-de-teste", auth="auto", timeout=10):
        srv = _Servidor(respostas)
        p = agent.Provedor(id="gemini", nome="Gemini", base_url=f"{srv.base}/v1beta/openai/",
                           var_chave="GEMINI_API_KEY", modelo=modelo, extra_params={"reasoning_effort": "low"},
                           auth=auth, url_nativa=f"{srv.base}/v1beta")
        monkeypatch.setenv("GEMINI_API_KEY", chave)
        provedores = {"gemini": p, "openrouter": agent.replace(agent.PROVEDORES["openrouter"],
                                                               base_url=f"{srv.base}/api/v1/")}
        monkeypatch.setattr(agent, "PROVEDORES", provedores)
        monkeypatch.setattr(agent, "provedor_ativo", lambda: p)
        monkeypatch.setattr(agent, "get_client", lambda pid: OpenAI(
            base_url=provedores[pid].base_url, api_key=provedores[pid].chave(), max_retries=0, timeout=timeout))
        return srv

    yield montar
    agent._AJUSTES.clear()


def _pergunta(df, texto="receita total?"):
    return agent.executar_agente(df, [{"role": "user", "content": texto}])


def test_api_nativa_com_x_goog_api_key_e_assinatura(df, gemini):
    srv = gemini(auth="nativo", respostas=[
        _nativo({"functionCall": {"name": "obter_receita_total", "args": {"categoria": "Todas"}},
                 "thoughtSignature": "SIG"}),
        _nativo({"text": "R$ 600,00"}),
    ])
    assert _pergunta(df) == "R$ 600,00"

    req1, req2 = srv.requisicoes
    assert req1["path"].endswith("/v1beta/models/gemini-3.8-flash:generateContent")
    assert req1["headers"]["x-goog-api-key"] == "AQ.chave-de-teste" and "authorization" not in req1["headers"]
    assert "Data Pilot" in req1["corpo"]["systemInstruction"]["parts"][0]["text"]
    assert req1["corpo"]["generationConfig"] == {"thinkingConfig": {"thinkingLevel": "low"}}
    decl = req1["corpo"]["tools"][0]["functionDeclarations"]
    assert {d["name"] for d in decl} == agent.TOOL_NAMES
    assert decl[0]["parameters"]["type"] == "OBJECT"
    assert decl[0]["parameters"]["properties"]["categoria"]["type"] == "STRING"
    # 2ª chamada: assinatura devolvida na mesma parte e resultado calculado pelo Python
    modelo, resposta_tool = req2["corpo"]["contents"][-2:]
    assert modelo["role"] == "model" and modelo["parts"][0]["thoughtSignature"] == "SIG"
    assert resposta_tool["parts"][0]["functionResponse"] == {"name": "obter_receita_total",
                                                             "response": {"result": 600.0}}


def test_chamadas_paralelas_voltam_no_mesmo_turno(df, gemini):
    srv = gemini(auth="nativo", respostas=[
        _nativo({"functionCall": {"name": "obter_total_pedidos", "args": {}}, "thoughtSignature": "S"},
                {"functionCall": {"name": "obter_ticket_medio", "args": {}}}),
        _nativo({"text": "ok"}),
    ])
    assert _pergunta(df) == "ok"
    respostas = srv.requisicoes[1]["corpo"]["contents"][-1]["parts"]
    assert [r["functionResponse"]["response"]["result"] for r in respostas] == [3, 200.0]


def test_modelo_sem_raciocinio_nao_recebe_thinking(df, gemini):
    srv = gemini([_nativo({"text": "ok"})], modelo="gemini-1.5-flash", auth="nativo")
    assert _pergunta(df) == "ok"
    assert "generationConfig" not in srv.requisicoes[0]["corpo"]


def test_thinking_recusado_repete_sem_ele(df, gemini):
    srv = gemini([_erro(400, "Thinking is not supported for this model."), _nativo({"text": "ok"})], auth="nativo")
    assert _pergunta(df) == "ok"
    assert "generationConfig" in srv.requisicoes[0]["corpo"]
    assert "generationConfig" not in srv.requisicoes[1]["corpo"]


def test_auto_usa_bearer_no_endpoint_compativel(df, gemini):
    srv = gemini([_compat("ok")])
    assert _pergunta(df) == "ok"
    req = srv.requisicoes[0]
    assert req["path"].endswith(COMPAT) and req["headers"]["authorization"] == "Bearer AQ.chave-de-teste"
    assert "x-goog-api-key" not in req["headers"]
    assert agent._AJUSTES["gemini"]["auth"] == "bearer"


def test_auth_key_recusada_no_compativel_tenta_nativo(df, gemini):
    # Mensagem real do endpoint compatível para chave AQ. recusada
    srv = gemini([_erro(400, "Invalid Auth key.", lista=True), _nativo({"text": "ok"})])
    assert _pergunta(df) == "ok"
    assert srv.requisicoes[0]["path"].endswith(COMPAT) and srv.requisicoes[1]["path"].endswith(NATIVO)
    assert srv.requisicoes[1]["headers"]["x-goog-api-key"] == "AQ.chave-de-teste"
    assert agent._AJUSTES["gemini"]["auth"] == "nativo"


def test_chave_recusada_nos_dois_mostra_detalhe(df, gemini):
    gemini([_erro(400, "Invalid Auth key.", lista=True),
            _erro(401, "API key not valid. Please pass a valid API key.", "UNAUTHENTICATED")])
    with pytest.raises(RuntimeError, match="Chave recusada.*API key not valid"):
        _pergunta(df)


def test_modelo_inexistente_lista_modelos_da_chave(df, gemini):
    gemini([_erro(404, "models/gemini-1.5-flash is not found for API version v1beta.", "NOT_FOUND")],
           modelo="gemini-1.5-flash", auth="nativo")
    with pytest.raises(RuntimeError) as exc:
        _pergunta(df)
    msg = str(exc.value)
    assert "not found" in msg and "gemini-3.6-flash, gemini-3.8-flash" in msg and "tts" not in msg


@pytest.mark.parametrize("status, trecho", [(400, "HTTP 400: INVALID_ARGUMENT: campo x"),
                                            (403, "Chave recusada")])
def test_erros_mostram_mensagem_real(df, gemini, status, trecho):
    gemini([_erro(status, "campo x inválido", "PERMISSION_DENIED" if status == 403 else "INVALID_ARGUMENT")],
           auth="bearer")
    with pytest.raises(RuntimeError, match=trecho):
        _pergunta(df)


ALTA_DEMANDA = "This model is currently experiencing high demand. Please try again later."


def test_sobrecarga_troca_de_modelo_e_memoriza(df, gemini):
    srv = gemini([_erro(503, ALTA_DEMANDA, "UNAVAILABLE", lista=True), _compat("ok"), _compat("de novo")])
    passos = []
    assert agent.executar_agente(df, [{"role": "user", "content": "x"}], on_step=passos.append) == "ok"
    modelos = [r["corpo"]["model"] for r in srv.requisicoes]
    assert modelos == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]  # Lite primeiro; preview/latest/tts ignorados
    assert any("gemini-3.8-flash indisponível; tentando gemini-3.5-flash-lite" in p for p in passos)
    # próxima pergunta já começa no modelo que funcionou
    assert _pergunta(df) == "de novo"
    assert srv.requisicoes[-1]["corpo"]["model"] == "gemini-3.5-flash-lite"


def test_falha_no_meio_refaz_pergunta_do_zero(df, gemini):
    tool_call = {"id": "c1", "type": "function", "function": {"name": "obter_total_pedidos", "arguments": "{}"},
                 "extra_content": {"google": {"thought_signature": "SIG-3.8"}}}
    srv = gemini([
        (200, {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
               "choices": [{"index": 0, "finish_reason": "tool_calls",
                            "message": {"role": "assistant", "content": None, "tool_calls": [tool_call]}}]}),
        _erro(503, ALTA_DEMANDA, "UNAVAILABLE", lista=True),
        _compat("ok"),
    ])
    assert _pergunta(df) == "ok"
    recomeco = srv.requisicoes[2]["corpo"]
    assert recomeco["model"] == "gemini-3.5-flash-lite"
    assert [m["role"] for m in recomeco["messages"]] == ["system", "user"]  # sem assinatura do outro modelo


def test_todos_sobrecarregados_lista_tentativas(df, gemini):
    srv = gemini([_erro(503, ALTA_DEMANDA, "UNAVAILABLE", lista=True)] * 4)
    with pytest.raises(agent.ModeloIndisponivel, match="Modelos tentados: gemini-3.8-flash, gemini-3.5-flash-lite, "
                                                        "gemini-3.6-flash, gemini-2.5-flash"):
        _pergunta(df)
    assert len(srv.requisicoes) == agent.MAX_TENTATIVAS


def test_orcamento_de_tempo_interrompe_tentativas(df, gemini, monkeypatch):
    srv = gemini([_erro(503, ALTA_DEMANDA, "UNAVAILABLE", lista=True)] * 4)
    monkeypatch.setattr(agent, "ORCAMENTO_S", 0)
    with pytest.raises(agent.ModeloIndisponivel):
        _pergunta(df)
    assert len(srv.requisicoes) == 1


def test_limite_de_uso_e_timeout_tambem_trocam_de_modelo(df, gemini):
    srv = gemini([_erro(429, "Resource has been exhausted.", "RESOURCE_EXHAUSTED"),
                  (*_compat("lento"), 1.5), _compat("ok")], timeout=0.5)
    assert _pergunta(df) == "ok"
    assert [r["corpo"]["model"] for r in srv.requisicoes] == ["gemini-3.8-flash", "gemini-3.5-flash-lite",
                                                              "gemini-3.6-flash"]


def test_gemini_sobrecarregado_usa_openrouter_de_reserva(df, gemini, monkeypatch):
    srv = gemini([_erro(503, ALTA_DEMANDA, "UNAVAILABLE", lista=True), _compat("via openrouter"),
                  _compat("de novo")])
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-teste")
    assert _pergunta(df) == "via openrouter"
    reserva = srv.requisicoes[1]
    assert reserva["path"] == "/api/v1/chat/completions" and reserva["corpo"]["model"] == "openrouter/free"
    assert reserva["headers"]["authorization"] == "Bearer sk-or-teste"
    assert "reasoning_effort" not in reserva["corpo"]
    assert _pergunta(df) == "de novo"  # próxima pergunta começa no provedor que funcionou
    assert srv.requisicoes[2]["path"] == "/api/v1/chat/completions"


def test_reserva_com_erro_de_configuracao_nao_esconde_o_motivo(df, gemini, monkeypatch):
    gemini([_erro(503, ALTA_DEMANDA, "UNAVAILABLE", lista=True), _erro(401, "No auth credentials found")])
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-invalida")
    with pytest.raises(agent.ModeloIndisponivel, match="No auth credentials found.*gemini-3.8-flash, openrouter/free"):
        _pergunta(df)


def test_alternativos_configurados_no_env(df, gemini, monkeypatch):
    srv = gemini([_erro(503, ALTA_DEMANDA, "UNAVAILABLE", lista=True), _compat("ok")])
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-escolhido")
    assert _pergunta(df) == "ok"
    assert srv.requisicoes[1]["corpo"]["model"] == "gemini-escolhido"


def test_alternativos_desativados(df, gemini, monkeypatch):
    gemini([_erro(503, ALTA_DEMANDA, "UNAVAILABLE", lista=True)])
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    with pytest.raises(agent.ModeloIndisponivel, match="indisponível no momento.*high demand"):
        _pergunta(df)
