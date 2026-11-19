"""Camada do agente: tool calling sem framework, via SDK da OpenAI (API compatível).

Provedor padrão: OpenRouter com o roteador gratuito `openrouter/free` (filtra modelos com suporte a tools).
Alternativa: Google Gemini pelo endpoint compatível com OpenAI.
"""
import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Callable

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import Omit, OpenAI
from openai.types.chat import ChatCompletion

import metrics

# Caminho explícito: independe do diretório de trabalho do launcher (IDE/terminal)
load_dotenv(Path(__file__).with_name(".env"))

PLACEHOLDER = "sua_chave_aqui"
CATEGORIA_TODAS = {"todas", "todas as categorias", "todos", "geral", "all"}
TIMEOUT_S = float(os.getenv("LLM_TIMEOUT", "20"))  # por chamada; sem isso o SDK espera até 600 s
ORCAMENTO_S = float(os.getenv("LLM_ORCAMENTO", "60"))  # tempo total para tentar modelos alternativos
MAX_TENTATIVAS = 4  # modelo principal + alternativos (outros modelos e/ou outro provedor)


@dataclass(frozen=True)
class Provedor:
    id: str
    nome: str
    base_url: str
    var_chave: str
    modelo: str
    extra_params: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)
    # bearer: endpoint compatível com OpenAI (Authorization: Bearer)
    # nativo: API nativa do Gemini (generateContent) com x-goog-api-key
    # auto:   tenta os dois e memoriza o que funcionar
    auth: str = "bearer"
    url_nativa: str = ""

    def chave(self) -> str:
        return os.getenv(self.var_chave, "").strip().strip('"').strip("'")

    def modos_auth(self) -> list[str]:
        modo = "nativo" if self.auth == "header" else self.auth
        if modo != "auto":
            return [modo]
        # O endpoint compatível exige "Authorization: Bearer" (sem ele: "Missing or invalid Authorization header")
        # e reconhece chaves AIza e AQ. ("Auth key", padrão do AI Studio desde 28/05/2026). A API nativa com
        # x-goog-api-key fica como segunda opção, caso a chave seja recusada no endpoint compatível.
        return ["bearer", "nativo"] if self.url_nativa else ["bearer"]

    def headers_requisicao(self, modo: str) -> dict:
        # Omit() remove o "Authorization: Bearer" que o SDK enviaria (duas credenciais = HTTP 400)
        if modo == "nativo":
            return {"Authorization": Omit(), "x-goog-api-key": self.chave()}
        return {}


PROVEDORES = {
    "openrouter": Provedor(
        id="openrouter",
        nome="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        var_chave="OPENROUTER_API_KEY",
        modelo=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        headers={"X-Title": "Data Pilot"},
    ),
    "gemini": Provedor(
        id="gemini",
        nome="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        var_chave="GEMINI_API_KEY",
        modelo=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        # reasoning_effort só vai para modelos com raciocínio (2.5+); ver _parametros().
        # Temperatura fica no padrão: no Gemini 3, abaixo de 1.0 pode entrar em loop.
        extra_params={"reasoning_effort": os.getenv("GEMINI_REASONING", "low")} if os.getenv(
            "GEMINI_REASONING", "low").strip() else {},
        auth=os.getenv("GEMINI_AUTH", "auto").strip().lower(),
        url_nativa="https://generativelanguage.googleapis.com/v1beta",
    ),
}


def _chave_valida(var: str) -> bool:
    valor = os.getenv(var, "").strip().strip('"').strip("'")
    return bool(valor) and valor != PLACEHOLDER


def provedor_ativo() -> Provedor:
    """LLM_PROVIDER explícito; senão, o primeiro provedor com chave configurada (OpenRouter tem prioridade)."""
    escolhido = os.getenv("LLM_PROVIDER", "").strip().lower()
    if escolhido:
        if escolhido not in PROVEDORES:
            raise RuntimeError(f"LLM_PROVIDER inválido: '{escolhido}'. Use: {', '.join(PROVEDORES)}")
        return PROVEDORES[escolhido]
    for p in PROVEDORES.values():
        if _chave_valida(p.var_chave):
            return p
    return PROVEDORES["openrouter"]


@st.cache_resource
def get_client(nome_provedor: str) -> OpenAI:
    # Lazy: o painel carrega mesmo sem chave; só o chat depende dela
    p = PROVEDORES[nome_provedor]
    if not _chave_valida(p.var_chave):
        raise RuntimeError(f"{p.var_chave} não configurada no arquivo .env")
    return OpenAI(base_url=p.base_url, api_key=p.chave(),
                  timeout=TIMEOUT_S, max_retries=0, default_headers=p.headers or None)


# ---------- Ferramentas (nome, descrição e parâmetros de cada função de metrics.py) ----------
_FILTROS = {
    "categoria": {"type": "string", "description": "Categoria para filtrar (opcional). Omitir para todas."},
    "data_inicio": {"type": "string", "description": "Data inicial inclusiva, formato AAAA-MM-DD (opcional)."},
    "data_fim": {"type": "string", "description": "Data final inclusiva, formato AAAA-MM-DD (opcional)."},
}


def _tool(nome: str, descricao: str, props: dict | None = None, obrigatorios: list[str] | None = None) -> dict:
    params = {"type": "object", "properties": {**(props or {}), **_FILTROS}}
    if obrigatorios:
        params["required"] = obrigatorios
    return {"type": "function", "function": {"name": nome, "description": descricao, "parameters": params}}


TOOLS = [
    _tool("obter_receita_total", "Receita total em R$ dos pedidos concluídos."),
    _tool("obter_total_pedidos", "Quantidade de pedidos concluídos."),
    _tool("obter_ticket_medio", "Ticket médio em R$ (receita / pedidos concluídos)."),
    _tool("obter_clientes_unicos", "Quantidade de clientes distintos com pedidos concluídos."),
    _tool("obter_receita_media_por_cliente", "Receita média por cliente em R$ (receita / clientes únicos)."),
    _tool("obter_taxa_cancelamento",
          "Taxa de cancelamento em % (cancelados / total de pedidos de qualquer status)."),
    _tool("obter_top_cidades", "Ranking das cidades com maior receita em R$ (pedidos concluídos).",
          {"limite": {"type": "integer", "description": "Quantidade de cidades (padrão 5)."}}),
    _tool("obter_receita_mensal", "Receita em R$ por mês (chave AAAA-MM) dos pedidos concluídos."),
    _tool("obter_resumo_por",
          "Tabela com todas as métricas (receita, pedidos, ticket médio, clientes, receita por cliente, "
          "cancelamento) agrupadas por categoria, cidade ou mês. Use para comparações e quebras: uma única chamada.",
          {"dimensao": {"type": "string", "enum": list(metrics.DIMENSOES), "description": "Agrupamento."}},
          ["dimensao"]),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

SYSTEM_PROMPT = (
    "Você é o Data Pilot, assistente de análise de vendas. Você não calcula nada: obtenha todo número pelas "
    "ferramentas e só explique o resultado. Para comparações por categoria, cidade ou mês, use obter_resumo_por. "
    "Se precisar de várias ferramentas, chame todas de uma vez. Para períodos, converta para data_inicio/data_fim "
    "(AAAA-MM-DD). Se a pergunta não puder ser respondida pelas ferramentas, diga isso. Responda em português do "
    "Brasil, de forma direta, no padrão brasileiro (R$ 1.234,56; 12,53%; DD/MM/AAAA), "
    "com tabela markdown para listas.\n"
    "Hoje: {hoje}. Período dos dados: {inicio} a {fim}. Categorias: {categorias}."
)


def _executar_tool(df, nome: str, argumentos: str | None) -> str:
    """Executa a função pedida pelo modelo, pelo nome, restrita às ferramentas declaradas."""
    if nome not in TOOL_NAMES:
        return f"Erro: ferramenta '{nome}' não disponível."
    try:
        args = json.loads(argumentos or "{}")
        args = {k: v for k, v in args.items() if v not in (None, "")}
        # Modelo às vezes manda categoria="Todas" para o total geral
        if str(args.get("categoria", "")).strip().lower() in CATEGORIA_TODAS:
            args.pop("categoria")
        resultado = getattr(metrics, nome)(df, **args)
        return json.dumps(resultado, ensure_ascii=False) if isinstance(resultado, dict) else str(resultado)
    except Exception as e:  # erro volta ao modelo como texto, sem derrubar o chat
        return f"Erro ao executar '{nome}': {e}"


def _fmt_args(argumentos: str | None) -> str:
    try:
        return ", ".join(f"{k}={v}" for k, v in json.loads(argumentos or "{}").items())
    except ValueError:
        return argumentos or ""


class ModeloIndisponivel(RuntimeError):
    """Erro transitório do modelo (sobrecarga 5xx, limite 429 ou tempo esgotado): vale tentar outro modelo."""


_TRANSITORIOS = (openai.InternalServerError, openai.RateLimitError, openai.APITimeoutError)

_AJUSTES: dict[str, dict] = {}  # aprendido em execução, por provedor: modo de auth e suporte a raciocínio


def _detalhe(e: Exception) -> str:
    """Mensagem real do provedor. O Gemini devolve erros como lista ([{"error": {...}}]),
    formato que o SDK não lê; sem isso só aparece "Error code: 400"."""
    corpo = getattr(e, "body", None)
    if isinstance(corpo, list) and corpo:
        corpo = corpo[0]
    if isinstance(corpo, dict):
        erro = corpo.get("error", corpo)
        if isinstance(erro, dict) and erro.get("message"):
            status = f"{erro.get('status')}: " if erro.get("status") else ""
            return f"{status}{erro['message']}"[:400]
    return str(getattr(e, "message", e))[:400]


def _suporta_raciocinio(modelo: str) -> bool:
    """reasoning_effort existe a partir do Gemini 2.5; nos anteriores gera HTTP 400."""
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", modelo)
    return not m or (int(m.group(1)), int(m.group(2) or 0)) >= (2, 5)


def _parametros(p: Provedor) -> dict:
    params = dict(p.extra_params)
    if p.id == "gemini" and (not _suporta_raciocinio(p.modelo) or _AJUSTES.get(p.id, {}).get("sem_raciocinio")):
        params.pop("reasoning_effort", None)
    return params


def _erro_de_chave(detalhe: str) -> bool:
    d = detalhe.lower()
    termos = ("api key", "api_key", "auth key", "credential", "authentication", "authorization",
              "unauthenticated", "token")
    return any(t in d for t in termos)


def _modelos_disponiveis(p: Provedor) -> list[str]:
    """Modelos Flash que a chave enxerga (para sugerir quando o modelo configurado falha)."""
    modo = _AJUSTES.get(p.id, {}).get("auth", p.modos_auth()[0])
    headers = p.headers_requisicao(modo)
    try:
        cliente = get_client(p.id).with_options(timeout=10, max_retries=0)
        if modo == "nativo":
            dados = cliente.get(f"{p.url_nativa}/models?pageSize=1000", cast_to=object, options={"headers": headers})
            ids = [m["name"] for m in dados.get("models", [])
                   if "generateContent" in m.get("supportedGenerationMethods", ["generateContent"])]
        else:
            ids = [m.id for m in cliente.models.list(**({"extra_headers": headers} if headers else {}))]
    except Exception:  # sugestão é melhor esforço
        return []
    ids = sorted(i.removeprefix("models/") for i in ids)
    return [i for i in ids if "flash" in i and not any(x in i for x in ("image", "tts", "live", "audio", "embed"))]


def _erro_amigavel(p: Provedor, e: Exception) -> str:
    """Traduz exceções do SDK em mensagens acionáveis, sempre com o detalhe real do provedor."""
    if isinstance(e, openai.APITimeoutError):
        return f"Modelo {p.modelo} não respondeu em {TIMEOUT_S:.0f}s (sobrecarga ou rede lenta)."
    if isinstance(e, openai.APIConnectionError):
        return f"Sem conexão com {p.base_url}. Verifique internet, proxy ou firewall."
    if isinstance(e, openai.RateLimitError):
        return f"Limite de uso do modelo {p.modelo} atingido. Aguarde alguns instantes e repita. ({_detalhe(e)})"
    detalhe = _detalhe(e)
    if isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError)) or (
            isinstance(e, openai.BadRequestError) and _erro_de_chave(detalhe)):
        return f"Chave recusada ({p.var_chave}). Confira o valor no .env ou gere outra. Detalhe: {detalhe}"
    if isinstance(e, openai.NotFoundError) or (isinstance(e, openai.BadRequestError) and "model" in detalhe.lower()):
        disponiveis = _modelos_disponiveis(p)
        dica = f" Modelos Flash liberados para a sua chave: {', '.join(disponiveis)}." if disponiveis else ""
        return f"Modelo '{p.modelo}' recusado pelo {p.nome}: {detalhe}.{dica} Ajuste o modelo no .env."
    if isinstance(e, openai.InternalServerError):
        return f"Modelo {p.modelo} indisponível no momento (HTTP {e.status_code}: {detalhe})"
    if isinstance(e, openai.APIStatusError):
        return f"{p.nome} retornou HTTP {e.status_code}: {detalhe}"
    return f"{type(e).__name__}: {e}"


# ---------- API nativa do Gemini (formato OpenAI <-> generateContent) ----------
def _schema_gemini(schema: dict) -> dict:
    """JSON Schema das tools -> Schema do Gemini (tipos em maiúsculas)."""
    out = {}
    for k, v in schema.items():
        if k == "type":
            out[k] = str(v).upper()
        elif k == "properties":
            out[k] = {nome: _schema_gemini(prop) for nome, prop in v.items()}
        elif k == "items":
            out[k] = _schema_gemini(v)
        else:
            out[k] = v
    return out


def _json_ou_texto(conteudo: str):
    try:
        return json.loads(conteudo)
    except (TypeError, ValueError):
        return conteudo


def _para_gemini(mensagens: list) -> tuple[dict | None, list]:
    """Mensagens no formato OpenAI -> (systemInstruction, contents) da API nativa."""
    sistema, contents = None, []
    for m in mensagens:
        papel = m["role"]
        if papel == "system":
            sistema = {"parts": [{"text": m["content"]}]}
        elif papel == "tool":
            parte = {"functionResponse": {"name": m["name"], "response": {"result": _json_ou_texto(m["content"])}}}
            ultimo = contents[-1] if contents else None
            if ultimo and ultimo["role"] == "user" and "functionResponse" in ultimo["parts"][0]:
                ultimo["parts"].append(parte)  # respostas de ferramentas paralelas no mesmo turno
            else:
                contents.append({"role": "user", "parts": [parte]})
        else:
            partes = [{"text": m["content"]}] if m.get("content") else []
            for tc in m.get("tool_calls") or []:
                parte = {"functionCall": {"name": tc["function"]["name"],
                                          "args": json.loads(tc["function"].get("arguments") or "{}")}}
                assinatura = (tc.get("extra_content") or {}).get("google", {}).get("thought_signature")
                if assinatura:  # Gemini 3 exige a assinatura de volta na mesma parte
                    parte["thoughtSignature"] = assinatura
                partes.append(parte)
            contents.append({"role": "model" if papel == "assistant" else "user", "parts": partes or [{"text": ""}]})
    return sistema, contents


def _thinking_nativo(modelo: str, esforco: str) -> dict:
    m = re.search(r"gemini-(\d+)", modelo)
    if m and int(m.group(1)) >= 3:
        return {"thinkingLevel": esforco}
    return {"thinkingBudget": {"low": 1024, "medium": 8192, "high": 24576}.get(esforco, 1024)}


def _chamar_nativo(p: Provedor, mensagens_llm: list, tool_choice: str, usar_tools: bool, params: dict):
    sistema, contents = _para_gemini(mensagens_llm)
    corpo: dict = {"contents": contents}
    if sistema:
        corpo["systemInstruction"] = sistema
    if usar_tools:
        corpo["tools"] = [{"functionDeclarations": [
            {"name": f["name"], "description": f["description"], "parameters": _schema_gemini(f["parameters"])}
            for f in (t["function"] for t in TOOLS)]}]
        corpo["toolConfig"] = {"functionCallingConfig": {"mode": "NONE" if tool_choice == "none" else "AUTO"}}
    if params.get("reasoning_effort"):
        corpo["generationConfig"] = {"thinkingConfig": _thinking_nativo(p.modelo, params["reasoning_effort"])}

    modelo = p.modelo.removeprefix("models/")
    dados = get_client(p.id).post(f"{p.url_nativa}/models/{modelo}:generateContent", cast_to=object, body=corpo,
                                  options={"headers": p.headers_requisicao("nativo")})

    candidatos = dados.get("candidates") or []
    if not candidatos:
        motivo = (dados.get("promptFeedback") or {}).get("blockReason", "resposta vazia")
        raise RuntimeError(f"{p.nome} não respondeu ({motivo}). Reformule a pergunta.")
    partes = (candidatos[0].get("content") or {}).get("parts") or []
    texto = "".join(pt.get("text", "") for pt in partes if not pt.get("thought"))
    chamadas = []
    for i, pt in enumerate(partes):
        if "functionCall" not in pt:
            continue
        fc = pt["functionCall"]
        chamada = {"id": fc.get("id") or f"call_{i}", "type": "function",
                   "function": {"name": fc["name"], "arguments": json.dumps(fc.get("args") or {}, ensure_ascii=False)}}
        if pt.get("thoughtSignature"):
            chamada["extra_content"] = {"google": {"thought_signature": pt["thoughtSignature"]}}
        chamadas.append(chamada)
    mensagem = {"role": "assistant", "content": texto or None, **({"tool_calls": chamadas} if chamadas else {})}
    return ChatCompletion.model_validate({
        "id": dados.get("responseId", "gemini"), "object": "chat.completion", "created": int(time.time()),
        "model": dados.get("modelVersion", modelo),
        "choices": [{"index": 0, "finish_reason": "tool_calls" if chamadas else "stop", "message": mensagem}],
    })


def _chamar_llm(p: Provedor, mensagens_llm: list, tool_choice: str, usar_tools: bool = True):
    """Chama o LLM com autenticação e parâmetros adaptativos:
    - auth "auto" (Gemini): endpoint compatível com Bearer e, se a chave for recusada, API nativa com
      x-goog-api-key; memoriza o que funcionou;
    - se o modelo recusar reasoning_effort, repete sem ele (e memoriza)."""
    estado = _AJUSTES.setdefault(p.id, {})
    modos = [estado["auth"]] if "auth" in estado else p.modos_auth()
    ultimo_erro: Exception | None = None
    for modo in modos:
        for _ in range(2):
            params = _parametros(p)
            try:
                if modo == "nativo":
                    resposta = _chamar_nativo(p, mensagens_llm, tool_choice, usar_tools, params)
                else:
                    extra = {"tools": TOOLS, "tool_choice": tool_choice} if usar_tools else {}
                    resposta = get_client(p.id).chat.completions.create(
                        model=p.modelo, messages=mensagens_llm, **extra, **params)
                estado["auth"] = modo
                return resposta
            except openai.BadRequestError as e:
                detalhe = _detalhe(e).lower()
                if "reasoning_effort" in params and ("reasoning" in detalhe or "thinking" in detalhe):
                    estado["sem_raciocinio"] = True
                    continue
                if not _erro_de_chave(detalhe):
                    raise RuntimeError(_erro_amigavel(p, e)) from None
                ultimo_erro = e
                break
            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                ultimo_erro = e
                break
            except _TRANSITORIOS as e:
                raise ModeloIndisponivel(_erro_amigavel(p, e)) from None
            except openai.OpenAIError as e:
                raise RuntimeError(_erro_amigavel(p, e)) from None
    raise RuntimeError(_erro_amigavel(p, ultimo_erro)) from None


def _ordenar_alternativos(ids: list[str]) -> list[str]:
    """Modelos estáveis: Lite primeiro (mais capacidade livre no plano gratuito), depois do mais novo ao mais antigo."""
    def versao(m: str) -> tuple[int, int]:
        v = re.search(r"gemini-(\d+)(?:\.(\d+))?", m)
        return (int(v.group(1)), int(v.group(2) or 0)) if v else (0, 0)
    estaveis = [m for m in ids if not any(x in m for x in ("preview", "exp", "latest"))]
    return sorted(estaveis, key=lambda m: ("lite" not in m, [-x for x in versao(m)]))


def _modelos_alternativos(p: Provedor) -> list[str]:
    """<PROVEDOR>_FALLBACK_MODELS no .env (vazio desativa); no Gemini, sem a variável, usa os modelos
    Flash liberados para a chave (consultados uma vez, só quando o principal falhar)."""
    configurado = os.getenv(f"{p.id.upper()}_FALLBACK_MODELS")
    if configurado is not None:
        return [m.strip() for m in configurado.split(",") if m.strip()]
    if p.id != "gemini":
        return []
    estado = _AJUSTES.setdefault(p.id, {})
    if "alternativos" not in estado:
        estado["alternativos"] = _ordenar_alternativos(_modelos_disponiveis(p))
    return estado["alternativos"]


def _candidatos(p: Provedor):
    """Ordem de tentativa (gerada sob demanda): o que funcionou por último, o modelo configurado,
    os alternativos do mesmo provedor e, por fim, outros provedores com chave no .env."""
    preferido = _AJUSTES.get("_preferido")
    if preferido and preferido[0] in PROVEDORES and _chave_valida(PROVEDORES[preferido[0]].var_chave):
        yield replace(PROVEDORES[preferido[0]], modelo=preferido[1])
    yield p
    for m in _modelos_alternativos(p):
        yield replace(p, modelo=m)
    for outro in PROVEDORES.values():
        if outro.id != p.id and _chave_valida(outro.var_chave):
            yield outro
            for m in _modelos_alternativos(outro):
                yield replace(outro, modelo=m)


def _responder(p: Provedor, df, mensagens: list, max_iter: int, passo: Callable[[str], None]) -> str:
    """Loop de tool calling com um modelo fixo."""
    contexto = SYSTEM_PROMPT.format(
        hoje=f"{date.today():%d/%m/%Y}",
        inicio=f"{df['data'].min():%d/%m/%Y}", fim=f"{df['data'].max():%d/%m/%Y}",
        categorias=", ".join(sorted(df["categoria"].dropna().unique())),
    )
    mensagens_llm = [{"role": "system", "content": contexto}]
    mensagens_llm.extend({"role": m["role"], "content": m["content"]} for m in mensagens)

    for rodada in range(1, max_iter + 1):
        # Última rodada: proíbe novas ferramentas e força a resposta final
        tool_choice = "auto" if rodada < max_iter else "none"
        t0 = time.perf_counter()
        resposta = _chamar_llm(p, mensagens_llm, tool_choice)
        if not resposta.choices:
            raise ModeloIndisponivel(f"{p.nome} ({p.modelo}) retornou resposta vazia.")
        msg = resposta.choices[0].message
        passo(f"{p.modelo} respondeu em {time.perf_counter() - t0:.1f}s · rodada {rodada}")

        if not msg.tool_calls:
            return msg.content or "Não foi possível responder com as ferramentas disponíveis. Reformule a pergunta."

        # Reenvia a mensagem do assistente intacta (inclui extra_content/thought_signature exigido pelo Gemini 3)
        msg_dict = msg.model_dump(exclude_none=True)
        msg_dict.setdefault("content", "")
        mensagens_llm.append(msg_dict)
        # Todas as respostas de ferramenta após as chamadas (ordem intercalada gera erro em alguns provedores)
        for tc in msg.tool_calls:
            passo(f"Ferramenta: {tc.function.name}({_fmt_args(tc.function.arguments)})")
            mensagens_llm.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": _executar_tool(df, tc.function.name, tc.function.arguments),
            })

    return "Não foi possível concluir a análise. Reformule a pergunta."


def executar_agente(df, mensagens: list, max_iter: int = 4,
                    on_step: Callable[[str], None] | None = None) -> str:
    """Responde com o modelo configurado. Se ele estiver sobrecarregado (503/429/timeout), refaz a pergunta
    do zero no próximo candidato (assinaturas de raciocínio não valem entre modelos), dentro de um orçamento
    de tempo, e memoriza o que funcionou para as próximas perguntas."""
    passo = on_step or (lambda _m: None)
    inicio = time.perf_counter()
    tentados: list[str] = []
    vistos: set[tuple[str, str]] = set()
    ultimo_erro: Exception | None = None
    for cand in _candidatos(provedor_ativo()):
        if (cand.id, cand.modelo) in vistos:
            continue
        if tentados and (len(tentados) >= MAX_TENTATIVAS or time.perf_counter() - inicio > ORCAMENTO_S):
            break
        if tentados:
            passo(f"{tentados[-1]} indisponível; tentando {cand.modelo} ({cand.nome})")
        vistos.add((cand.id, cand.modelo))
        tentados.append(cand.modelo)
        try:
            resposta = _responder(cand, df, mensagens, max_iter, passo)
            _AJUSTES["_preferido"] = (cand.id, cand.modelo)
            return resposta
        except ModeloIndisponivel as e:
            ultimo_erro = e
        except RuntimeError as e:
            if len(tentados) == 1:  # problema de configuração do modelo principal: mostrar já
                raise
            ultimo_erro = e  # alternativo com problema (ex.: chave de outro provedor): segue a fila
    msg = f"{ultimo_erro} Modelos tentados: {', '.join(tentados)}. Tente novamente em instantes."
    raise ModeloIndisponivel(msg)


# ---------- Diagnóstico: python agent.py ----------
def diagnosticar() -> int:
    """Testa provedor, chave, modelo e uma pergunta real, com tempos. Nunca imprime a chave."""
    from data import carregar_dados

    p = provedor_ativo()
    chave = p.chave()
    formato = "AQ. (auth key)" if chave.startswith("AQ.") else "AIza (standard)" if chave.startswith("AIza") else \
        "sk-or (OpenRouter)" if chave.startswith("sk-or") else "desconhecido"
    print(f"Provedor: {p.nome} | modelo: {p.modelo} | {p.var_chave}: "
          f"{'definida' if _chave_valida(p.var_chave) else 'AUSENTE'} ({len(chave)} caracteres, formato {formato})")
    print(f"Autenticação: {p.auth} | raciocínio: {_parametros(p).get('reasoning_effort', 'não enviado')} "
          f"| timeout {TIMEOUT_S:.0f}s")
    if not _chave_valida(p.var_chave):
        print(f"FALHA: defina {p.var_chave} no .env")
        return 1

    t0 = time.perf_counter()
    try:
        r = _chamar_llm(p, [{"role": "user", "content": "Responda só: OK"}], "none", usar_tools=False)
        texto = (r.choices[0].message.content or "").strip()[:40]
        modo = _AJUSTES[p.id]["auth"]
        print(f"[1/3] Chave e modelo OK ({time.perf_counter() - t0:.1f}s, auth={modo}): {texto}")
    except ModeloIndisponivel as e:
        print(f"[1/3] AVISO ({time.perf_counter() - t0:.1f}s): {e} O agente tentará modelos alternativos.")
    except RuntimeError as e:
        print(f"[1/3] FALHA ({time.perf_counter() - t0:.1f}s): {e}")
        return 1

    disponiveis = _modelos_disponiveis(p)
    print(f"[2/3] Modelos Flash liberados para a chave: {', '.join(disponiveis) or 'lista indisponível'}")
    if p.id == "gemini":
        alternativos = [m for m in _modelos_alternativos(p) if m != p.modelo]
        print(f"      Alternativos em caso de sobrecarga: {', '.join(alternativos) or 'nenhum'}")
    reservas = [f"{o.nome} ({o.modelo})" for o in PROVEDORES.values() if o.id != p.id and _chave_valida(o.var_chave)]
    print(f"      Outro provedor de reserva: {', '.join(reservas) or 'nenhum (defina OPENROUTER_API_KEY no .env)'}")

    df = carregar_dados()
    t0 = time.perf_counter()
    try:
        resposta = executar_agente(df, [{"role": "user", "content": "Qual a receita total?"}],
                                   on_step=lambda m: print(f"      {time.perf_counter() - t0:.1f}s · {m}"))
    except RuntimeError as e:
        print(f"[3/3] FALHA ({time.perf_counter() - t0:.1f}s): {e}")
        return 1
    esperado = f"{metrics.obter_receita_total(df):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    usado = (_AJUSTES.get("_preferido") or (p.id, p.modelo))[1]
    print(f"[3/3] Agente OK com {usado} ({time.perf_counter() - t0:.1f}s). Valor correto: R$ {esperado}")
    print(f"      Resposta: {resposta[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(diagnosticar())
