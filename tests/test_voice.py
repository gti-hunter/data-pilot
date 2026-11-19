"""Testes da camada de voz sem rede e sem baixar modelos (motores simulados nas bordas)."""
import asyncio
import io
import json
import os
import sys
import types
import wave

import pytest

import voice


# ---------- texto_para_fala ----------
def test_tabela_vira_frases_e_markdown_some():
    resposta = (
        "**Receita por categoria** no 1º trimestre:\n\n"
        "| Categoria | Receita |\n|---|---|\n"
        "| Eletrônicos | R$ 411.698,05 |\n| Móveis | R$ 171.852,32 |\n\n"
        "- Total de [pedidos](http://x) concluídos."
    )
    falado = voice.texto_para_fala(resposta)
    assert falado == ("Receita por categoria no 1º trimestre. Eletrônicos: R$ 411.698,05. "
                      "Móveis: R$ 171.852,32. Total de pedidos concluídos.")


def test_texto_longo_corta_em_frase_e_avisa():
    falado = voice.texto_para_fala("Frase curta. " * 100, limite=60)
    assert len(falado) < 120 and falado.endswith("Os demais detalhes estão na tela.")
    assert voice.texto_para_fala("") == ""


# ---------- descoberta de motores ----------
def test_descoberta_usa_proximo_motor_quando_um_falha():
    class Ok:
        nome = "ok"

    def quebrado():
        raise ImportError("pacote ausente")

    motor = voice._descobrir({"quebrado": quebrado, "ok": Ok}, ("quebrado", "ok"), "", "voz")
    assert isinstance(motor, Ok)


def test_descoberta_sem_motor_explica_como_instalar():
    with pytest.raises(RuntimeError, match="quebrado: ImportError: pacote ausente.*pip install -r requirements.txt"):
        voice._descobrir({"quebrado": lambda: (_ for _ in ()).throw(ImportError("pacote ausente"))},
                         ("quebrado",), "", "transcrição")


# ---------- edge-tts (simulado) ----------
@pytest.fixture
def edge_falso(monkeypatch):
    chamadas = []

    class Communicate:
        def __init__(self, texto, voz, proxy=None):
            chamadas.append({"texto": texto, "voz": voz, "proxy": proxy})

        async def stream(self):
            yield {"type": "audio", "data": b"ID3"}
            yield {"type": "SentenceBoundary", "text": "x"}
            yield {"type": "audio", "data": b"mp3"}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=Communicate))
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    return chamadas


def test_edge_tts_junta_audio_e_usa_voz_e_proxy(edge_falso):
    audio = voice.EdgeTTS("pt-BR-FranciscaNeural").sintetizar("Olá")
    assert audio == voice.Audio(b"ID3mp3", "audio/mpeg")
    assert edge_falso == [{"texto": "Olá", "voz": "pt-BR-FranciscaNeural", "proxy": "http://proxy:8080"}]


def test_edge_tts_funciona_dentro_de_event_loop(edge_falso):
    async def dentro_do_loop():
        return voice.EdgeTTS().sintetizar("Olá")

    assert asyncio.run(dentro_do_loop()).dados == b"ID3mp3"


# ---------- faster-whisper (simulado) ----------
def test_whisper_transcreve_em_portugues_com_vocabulario(monkeypatch):
    recebido = {}

    class WhisperModel:
        def __init__(self, modelo, device, compute_type):
            recebido.update(modelo=modelo, device=device, compute_type=compute_type)

        def transcribe(self, audio, **kwargs):
            recebido.update(audio=audio.read(), **kwargs)
            segmentos = [types.SimpleNamespace(text=" Qual a receita"), types.SimpleNamespace(text=" de esportes? ")]
            return iter(segmentos), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WhisperModel))
    texto = voice.FasterWhisperSTT("base").transcrever(b"RIFF", ["Esportes", "Curitiba"])
    assert texto == "Qual a receita de esportes?"
    assert recebido["modelo"] == "base" and recebido["compute_type"] == "int8" and recebido["audio"] == b"RIFF"
    assert recebido["language"] == "pt" and recebido["vad_filter"] is True
    assert "Esportes, Curitiba" in recebido["initial_prompt"]


def _wav_16k(segundos=0.5) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * segundos))
    return buf.getvalue()


def test_wav_do_microfone_e_decodificado_pelo_faster_whisper_real():
    audio_mod = pytest.importorskip("faster_whisper.audio")
    amostras = audio_mod.decode_audio(io.BytesIO(_wav_16k(0.5)), sampling_rate=16000)
    assert len(amostras) == 8000


# ---------- fluxo no app: voz -> agente -> voz ----------
@pytest.fixture
def app_voz(monkeypatch):
    import streamlit as st
    from openai.types.chat import ChatCompletion
    from streamlit.testing.v1 import AppTest
    from streamlit.typing import ChatInputValue

    import agent

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(raiz)
    for var in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    agent._AJUSTES.clear()
    ferramentas = []

    def resp(conteudo=None, chamadas=None):
        msg = {"role": "assistant", "content": conteudo}
        if chamadas:
            msg["tool_calls"] = [
                {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                for i, (n, a) in enumerate(chamadas)]
        return ChatCompletion.model_validate({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                              "choices": [{"index": 0, "finish_reason": "stop", "message": msg}]})

    def llm_falso(p, mensagens, tool_choice, usar_tools=True):
        if mensagens[-1]["role"] == "user":
            return resp(chamadas=[("obter_receita_total", {"categoria": "esporte", "data_inicio": "2026-11-01",
                                                           "data_fim": "2026-11-30"})])
        ferramentas.append(mensagens[-1]["content"])
        return resp(f"A receita de Esportes em novembro foi R$ {mensagens[-1]['content']}.")

    monkeypatch.setattr(agent, "_chamar_llm", llm_falso)
    falas = []

    class Stt:
        def transcrever(self, audio, vocabulario=None):
            assert audio == b"RIFF-gravado" and "Esportes" in vocabulario
            return "Qual a receita de esporte em novembro?"

    monkeypatch.setattr(voice, "obter_transcricao", lambda preferido="": Stt())
    monkeypatch.setattr(voice, "responder_em_audio",
                        lambda texto: falas.append(texto) or voice.Audio(b"ID3-resposta", "audio/mpeg"))

    def montar(entrada):
        monkeypatch.setattr(st, "chat_input", lambda *a, **k: entrada)
        at = AppTest.from_file(os.path.join(raiz, "app.py"), default_timeout=60).run()
        at.sidebar.radio[0].set_value("Conversar com Dados").run()
        return at, ferramentas, falas

    gravacao = types.SimpleNamespace(getvalue=lambda: b"RIFF-gravado")
    return montar, ChatInputValue(text="", audio=gravacao, _include_audio=True)


def test_pergunta_por_voz_responde_em_texto_e_voz(app_voz):
    montar, entrada_voz = app_voz
    at, ferramentas, falas = montar(entrada_voz)
    assert not at.exception
    textos = [m.value for m in at.markdown]
    assert "Qual a receita de esporte em novembro?" in textos
    assert "Transcrito da sua voz" in [c.value for c in at.caption]
    assert ferramentas == ["18693.11"]  # número calculado pelo metrics.py, igual ao painel
    assert "A receita de Esportes em novembro foi R$ 18693.11." in textos
    assert falas == ["A receita de Esportes em novembro foi R$ 18693.11."]
    assert len(at.get("audio")) == 1


def test_pergunta_digitada_nao_gera_audio(app_voz):
    montar, _ = app_voz
    at, _, falas = montar("Qual a receita de esporte em novembro?")
    assert not at.exception and falas == [] and len(at.get("audio")) == 0


def test_falha_de_download_do_modelo_sugere_verificar_internet():
    def sem_rede():
        raise OSError("Cannot find an appropriate cached snapshot")

    with pytest.raises(RuntimeError, match="baixado do Hugging Face"):
        voice._descobrir({"faster-whisper": sem_rede}, ("faster-whisper",), "", "transcrição")


# ---------- conversa contínua (modo mãos livres) ----------
def test_ruido_e_comando_de_encerrar():
    assert voice.eh_ruido("") and voice.eh_ruido("Legendas pela comunidade Amara.org") and voice.eh_ruido("ok")
    assert not voice.eh_ruido("Qual a receita de esportes?")
    assert voice.pede_encerrar("Pode encerrar a conversa, obrigado") and not voice.pede_encerrar("Qual a receita?")


@pytest.fixture
def continua(app_voz, monkeypatch):
    import base64

    import voz_continua

    montar, _ = app_voz
    chamadas = []

    def preparar(falas):
        fila = iter(falas)

        def componente(turno, audio=None, parar=False, key="voz_continua"):
            chamadas.append({"turno": turno, "audio": audio, "parar": parar})
            return next(fila, None)

        monkeypatch.setattr(voz_continua, "conversa_por_voz", componente)
        return montar(None)

    fala = {"id": "f1", "wav": base64.b64encode(b"RIFF-gravado").decode()}
    return preparar, chamadas, fala


def test_conversa_continua_responde_e_devolve_audio_ao_navegador(continua):
    preparar, chamadas, fala = continua
    at, ferramentas, falas = preparar([fala])
    assert not at.exception
    assert ferramentas == ["18693.11"] and len(falas) == 1
    # após responder, a página recarrega e o componente recebe o próximo turno com o áudio para tocar
    assert chamadas[-1]["turno"] == 1 and chamadas[-1]["audio"]["mime"] == "audio/mpeg"
    assert "Qual a receita de esporte em novembro?" in [m.value for m in at.markdown]
    assert len(at.get("audio")) == 1  # resposta fica no histórico (sem tocar de novo)


def test_ruido_so_volta_a_ouvir(continua, monkeypatch):
    preparar, chamadas, fala = continua
    monkeypatch.setattr(voice, "obter_transcricao",
                        lambda preferido="": types.SimpleNamespace(transcrever=lambda a, v=None: "Obrigado."))
    at, ferramentas, falas = preparar([fala])
    assert not at.exception and ferramentas == [] and falas == []
    assert chamadas[-1] == {"turno": 1, "audio": None, "parar": False}


def test_comando_de_voz_encerra_a_conversa(continua, monkeypatch):
    preparar, chamadas, fala = continua
    monkeypatch.setattr(voice, "obter_transcricao",
                        lambda preferido="": types.SimpleNamespace(transcrever=lambda a, v=None: "Encerrar conversa."))
    at, ferramentas, _ = preparar([fala])
    assert not at.exception and ferramentas == []
    assert chamadas[-1] == {"turno": 1, "audio": None, "parar": True}
    assert "Conversa por voz encerrada." in [i.value for i in at.info]
