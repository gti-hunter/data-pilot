"""Data Pilot: painel de vendas + conversa com os dados. Execute com: streamlit run app.py"""
import base64
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

import agent
import voice
import voz_continua
from data import ARQUIVO_CRU, ARQUIVO_LIMPO, carregar_dados
from metrics import (
    obter_clientes_unicos, obter_receita_media_por_cliente, obter_receita_mensal, obter_receita_total,
    obter_taxa_cancelamento, obter_ticket_medio, obter_top_cidades, obter_total_pedidos,
)

if get_script_run_ctx() is None:
    raise SystemExit("O Data Pilot é um app Streamlit. Execute com: streamlit run app.py")

MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
COR = "#1f4e79"


# ---------- Formatação pt-BR ----------
def fmt_num(v: float, casas: int = 0) -> str:
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_brl(v: float) -> str:
    return f"R$ {fmt_num(v, 2)}"


def rotulo_mes(periodo: str) -> str:
    ano, mes = periodo.split("-")
    return f"{MESES[int(mes) - 1]}/{ano[2:]}"


def layout_brl(fig, titulo_x: str):
    # separators=",." -> decimal ',' e milhar '.' em eixos, rótulos e tooltips
    fig.update_layout(separators=",.", xaxis_title=titulo_x, yaxis_title="Receita (R$)",
                      xaxis_type="category", margin=dict(t=30))
    fig.update_yaxes(tickprefix="R$ ", tickformat=",.0f")
    fig.update_traces(hovertemplate="%{x}<br>R$ %{y:,.2f}<extra></extra>")
    return fig


# ---------- Páginas ----------
def pagina_painel(df: pd.DataFrame) -> None:
    st.title("Painel de Vendas")

    d_min, d_max = df["data"].min().date(), df["data"].max().date()
    f1, f2 = st.columns([1, 1])
    categorias = sorted(df["categoria"].dropna().unique().tolist())
    categoria_sel = f1.selectbox("Categoria", ["Todas"] + categorias)
    periodo = f2.date_input("Período", value=(d_min, d_max), min_value=d_min, max_value=d_max, format="DD/MM/YYYY")
    # Enquanto o usuário escolhe o intervalo, o widget devolve só a data inicial
    inicio, fim = (periodo if isinstance(periodo, tuple) and len(periodo) == 2 else (d_min, d_max))
    filtros = {
        "categoria": None if categoria_sel == "Todas" else categoria_sel,
        "data_inicio": inicio.isoformat(),
        "data_fim": fim.isoformat(),
    }

    st.caption(
        f"Fonte: {ARQUIVO_CRU} · Período: {inicio:%d/%m/%Y} a {fim:%d/%m/%Y} · Métricas sobre pedidos concluídos; "
        "a taxa de cancelamento considera todos os pedidos."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Receita Total", fmt_brl(obter_receita_total(df, **filtros)))
    c2.metric("Total de Pedidos", fmt_num(obter_total_pedidos(df, **filtros)))
    c3.metric("Ticket Médio", fmt_brl(obter_ticket_medio(df, **filtros)))
    c4, c5, c6 = st.columns(3)
    c4.metric("Clientes Únicos", fmt_num(obter_clientes_unicos(df, **filtros)))
    c5.metric("Receita Média por Cliente", fmt_brl(obter_receita_media_por_cliente(df, **filtros)))
    c6.metric("Taxa de Cancelamento", f"{fmt_num(obter_taxa_cancelamento(df, **filtros), 2)}%")

    st.divider()
    g1, g2 = st.columns(2)

    with g1:
        st.subheader("Receita Mensal")
        mensal = obter_receita_mensal(df, **filtros)
        if mensal:
            df_m = pd.DataFrame({"Mês": [rotulo_mes(p) for p in mensal], "Receita": list(mensal.values())})
            fig = px.line(df_m, x="Mês", y="Receita", markers=True, color_discrete_sequence=[COR])
            st.plotly_chart(layout_brl(fig, "Mês"), width="stretch")
        else:
            st.info("Sem pedidos concluídos no filtro selecionado.")

    with g2:
        st.subheader("Top 5 Cidades (Receita)")
        top = obter_top_cidades(df, 5, **filtros)
        if top:
            df_c = pd.DataFrame(list(top.items()), columns=["Cidade", "Receita"])
            fig = layout_brl(px.bar(df_c, x="Cidade", y="Receita", color_discrete_sequence=[COR]), "Cidade")
            fig.update_traces(texttemplate="R$ %{y:,.0f}", textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Sem pedidos concluídos no filtro selecionado.")


def _vocabulario(df: pd.DataFrame) -> list[str]:
    """Termos do negócio que ajudam o Whisper a acertar nomes de categorias e cidades."""
    return sorted(df["categoria"].dropna().unique()) + sorted(df["cidade"].dropna().unique())


def _ler_pergunta(df: pd.DataFrame) -> tuple[str | None, bool]:
    """Texto digitado ou áudio gravado no botão de microfone do próprio campo do chat.
    Retorna (pergunta, veio_por_voz)."""
    entrada = st.chat_input("Digite ou grave sua pergunta. Ex.: Qual a receita por categoria no 1º trimestre?",
                            accept_audio=voice.VOZ_ATIVA)
    if entrada is None:
        return None, False
    if isinstance(entrada, str):
        return entrada.strip() or None, False
    texto, audio = (entrada.text or "").strip(), getattr(entrada, "audio", None)
    if texto or audio is None:
        return texto or None, False
    with st.spinner("Transcrevendo sua pergunta..."):
        try:
            texto = voice.obter_transcricao().transcrever(audio.getvalue(), _vocabulario(df))
        except Exception as e:
            st.error(f"Não foi possível transcrever o áudio: {e}")
            return None, False
    if not texto:
        st.warning("Não entendi o áudio. Tente de novo, falando mais perto do microfone.")
        return None, False
    return texto, True


def _falar(texto: str, tocar: bool = True) -> voice.Audio | None:
    """Gera a resposta em voz. tocar=True toca aqui (pergunta avulsa pelo microfone do chat);
    no modo contínuo quem toca é o componente, que depois volta a ouvir. Falha na voz não bloqueia o texto."""
    try:
        with st.spinner("Gerando a resposta em voz..."):
            audio = voice.responder_em_audio(texto)
    except Exception as e:
        st.caption(f"Resposta em voz indisponível: {e}")
        return None
    if tocar:
        st.audio(audio.dados, format=audio.mime, autoplay=True)
    return audio


def _responder(df: pd.DataFrame, prompt: str, por_voz: bool, falar: bool, tocar: bool = True) -> voice.Audio | None:
    """Mostra a pergunta, consulta o agente e mostra (e opcionalmente fala) a resposta."""
    st.session_state.messages.append({"role": "user", "content": prompt, "voz": por_voz})
    with st.chat_message("user"):
        st.markdown(prompt)
        if por_voz:
            st.caption("Transcrito da sua voz")

    with st.chat_message("assistant"):
        status = st.status("Analisando dados...", expanded=False)
        t0 = time.perf_counter()
        try:
            resposta = agent.executar_agente(
                df, st.session_state.messages,
                on_step=lambda m: status.write(f"{time.perf_counter() - t0:.1f}s · {m}"),
            )
        except Exception as e:
            status.update(label=f"Falhou após {time.perf_counter() - t0:.1f}s", state="error")
            st.session_state.messages.pop()  # não deixa pergunta sem resposta no histórico
            st.error(f"Erro ao consultar o agente: {e}")
            if not tocar:  # modo contínuo: a página é recarregada em seguida, então guarda o aviso
                st.session_state.aviso_voz = f"Erro ao consultar o agente: {e}"
            return None
        status.update(label=f"Concluído em {time.perf_counter() - t0:.1f}s", state="complete")
        st.markdown(resposta)
        mensagem = {"role": "assistant", "content": resposta}
        audio = _falar(resposta, tocar=tocar) if falar else None
        if audio:
            mensagem["audio"] = audio
        st.session_state.messages.append(mensagem)
        return audio


def _conversa_continua(df: pd.DataFrame, responder_voz: bool) -> None:
    """Modo mãos livres: o componente ouve, detecta o fim da fala e envia; aqui transcreve, responde
    e devolve o áudio para o componente tocar e voltar a ouvir, sem novo clique."""
    estado = st.session_state.setdefault("voz", {"turno": 0, "audio": None, "parar": False, "ultima_fala": None})
    with st.sidebar:
        st.markdown("**Conversa por voz**")
        fala = voz_continua.conversa_por_voz(estado["turno"], estado["audio"], estado["parar"])
    estado["audio"], estado["parar"] = None, False  # já entregues ao navegador nesta execução
    if not fala or fala.get("id") == estado["ultima_fala"]:
        return
    estado["ultima_fala"] = fala["id"]
    try:
        with st.spinner("Transcrevendo..."):
            texto = voice.obter_transcricao().transcrever(base64.b64decode(fala["wav"]), _vocabulario(df))
    except Exception as e:
        texto = ""
        st.session_state.aviso_voz = f"Não foi possível transcrever o áudio: {e}"
    if voice.pede_encerrar(texto):
        estado["parar"] = True
        st.session_state.aviso_voz = "Conversa por voz encerrada."
    elif not voice.eh_ruido(texto):
        audio = _responder(df, texto, por_voz=True, falar=responder_voz, tocar=False)
        if audio:
            estado["audio"] = {"b64": base64.b64encode(audio.dados).decode(), "mime": audio.mime}
    estado["turno"] += 1  # avisa o navegador: toca a resposta (se houver) e volta a ouvir
    st.rerun()


def pagina_chat(df: pd.DataFrame) -> None:
    st.title("Conversar com Dados")
    st.caption("O Python calcula, a IA só explica: cada número vem de uma função do metrics.py chamada pelo modelo."
               + (" Para conversar por voz sem clicar a cada pergunta, use **Iniciar conversa por voz** na barra "
                  "lateral." if voice.VOZ_ATIVA else ""))

    if "messages" not in st.session_state:
        st.session_state.messages = []
    responder_voz = voice.VOZ_ATIVA and st.sidebar.toggle(
        "Responder em voz", value=True, help="Quando a pergunta for falada, a resposta também é falada.")
    if st.sidebar.button("Limpar conversa", width="stretch"):
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("voz"):
                st.caption("Transcrito da sua voz")
            if msg.get("audio"):
                st.audio(msg["audio"].dados, format=msg["audio"].mime)
    if aviso := st.session_state.pop("aviso_voz", None):
        st.info(aviso)

    if voice.VOZ_ATIVA:
        _conversa_continua(df, responder_voz)

    prompt, por_voz = _ler_pergunta(df)
    if prompt:
        _responder(df, prompt, por_voz, falar=por_voz and responder_voz)


# ---------- Main ----------
st.set_page_config(page_title="Data Pilot", layout="wide")

df = carregar_dados(ARQUIVO_LIMPO, ARQUIVO_CRU)
if df.empty:
    st.error(f"Dados não encontrados. Coloque o {ARQUIVO_CRU} na pasta do projeto.")
    st.stop()

st.sidebar.title("Data Pilot")
menu = st.sidebar.radio("Menu", ["Painel de Vendas", "Conversar com Dados"])
try:
    p = agent.provedor_ativo()
    st.sidebar.caption(f"LLM: {p.nome} · {p.modelo}")
except RuntimeError as e:
    st.sidebar.warning(str(e))

if menu == "Painel de Vendas":
    pagina_painel(df)
else:
    pagina_chat(df)
