"""Camada de voz: transcrição (fala -> texto) e síntese (texto -> fala).

Mesmo desenho do módulo speech/ do OpenJarvis: motores plugáveis com descoberta automática.
- Transcrição: faster-whisper local (a gravação da voz não sai do computador).
- Síntese: vozes neurais pt-BR via edge-tts (só o texto da resposta é enviado).
A voz é só entrada e saída: quem responde continua sendo o agente, com números do metrics.py.
Autoteste: python voice.py
"""
from __future__ import annotations

import asyncio
import io
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

VOZ_ATIVA = os.getenv("VOICE_ENABLED", "true").strip().lower() not in ("false", "0", "nao", "não", "off")
STT_MODELO = os.getenv("VOICE_STT_MODEL", "base")  # tiny | base | small | medium
TTS_VOZ = os.getenv("VOICE_TTS_VOICE", "pt-BR-FranciscaNeural")  # ou pt-BR-AntonioNeural
TTS_MAX_CARACTERES = int(os.getenv("VOICE_TTS_MAX_CHARS", "600"))
TTS_TIMEOUT_S = float(os.getenv("VOICE_TTS_TIMEOUT", "20"))
IDIOMA = "pt"


@dataclass(frozen=True)
class Audio:
    dados: bytes
    mime: str


class MotorTranscricao(Protocol):
    nome: str

    def transcrever(self, audio: bytes, vocabulario: list[str] | None = None) -> str: ...


class MotorVoz(Protocol):
    nome: str

    def sintetizar(self, texto: str) -> Audio: ...


# ---------- Motores ----------
class FasterWhisperSTT:
    """Whisper local (CTranslate2, CPU int8). O modelo é baixado uma vez do Hugging Face."""

    nome = "faster-whisper"

    def __init__(self, modelo: str = STT_MODELO, device: str = "cpu", compute_type: str = "int8") -> None:
        from faster_whisper import WhisperModel  # import tardio: só carrega se a voz for usada

        self._modelo = WhisperModel(modelo, device=device, compute_type=compute_type)

    def transcrever(self, audio: bytes, vocabulario: list[str] | None = None) -> str:
        # initial_prompt orienta o Whisper para os termos do negócio (categorias, cidades, métricas)
        prompt = "Perguntas sobre vendas: receita, pedidos, ticket médio, clientes únicos, taxa de cancelamento."
        if vocabulario:
            prompt += " " + ", ".join(vocabulario) + "."
        segmentos, _ = self._modelo.transcribe(
            io.BytesIO(audio), language=IDIOMA, vad_filter=True, initial_prompt=prompt)
        return " ".join(s.text.strip() for s in segmentos).strip()


class EdgeTTS:
    """Vozes neurais pt-BR do Microsoft Edge (gratuito, sem chave; não é um produto oficial com SLA)."""

    nome = "edge-tts"

    def __init__(self, voz: str = TTS_VOZ) -> None:
        import edge_tts

        self._edge_tts, self._voz = edge_tts, voz

    def sintetizar(self, texto: str) -> Audio:
        async def gerar() -> bytes:
            comunicador = self._edge_tts.Communicate(
                texto, self._voz, proxy=os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or None)
            partes = bytearray()
            async for pedaco in comunicador.stream():
                if pedaco["type"] == "audio":
                    partes.extend(pedaco["data"])
            return bytes(partes)

        dados = _rodar_async(asyncio.wait_for(gerar(), TTS_TIMEOUT_S))
        if not dados:
            raise RuntimeError("edge-tts não retornou áudio")
        return Audio(dados, "audio/mpeg")


def _rodar_async(corotina):
    """Executa uma corotina mesmo se já houver um event loop ativo na thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(corotina)
    resultado: dict = {}

    def alvo() -> None:
        try:
            resultado["ok"] = asyncio.run(corotina)
        except Exception as e:  # repassa o erro para a thread chamadora
            resultado["erro"] = e

    t = threading.Thread(target=alvo)
    t.start()
    t.join()
    if "erro" in resultado:
        raise resultado["erro"]
    return resultado["ok"]


# ---------- Descoberta (como speech/_discovery.py do OpenJarvis) ----------
MOTORES_STT: dict[str, Callable[[], MotorTranscricao]] = {"faster-whisper": FasterWhisperSTT}
MOTORES_TTS: dict[str, Callable[[], MotorVoz]] = {"edge-tts": EdgeTTS}
ORDEM_STT = ("faster-whisper",)
ORDEM_TTS = ("edge-tts",)


def _descobrir(motores: dict, ordem: tuple[str, ...], preferido: str, tipo: str):
    erros, falta_pacote = [], False
    for nome in dict.fromkeys((preferido, *ordem)):
        if not nome or nome not in motores:
            continue
        try:
            return motores[nome]()
        except Exception as e:  # motor indisponível: tenta o próximo
            falta_pacote |= isinstance(e, ImportError)
            erros.append(f"{nome}: {type(e).__name__}: {str(e)[:200]}")
    dica = ("Instale as dependências com: pip install -r requirements.txt" if falta_pacote or not erros else
            "Na primeira vez o modelo é baixado do Hugging Face: verifique a internet ou o proxy e tente de novo.")
    raise RuntimeError(f"Nenhum motor de {tipo} disponível ({'; '.join(erros) or 'nenhum registrado'}). {dica}")


@st.cache_resource(show_spinner="Carregando o modelo de voz (só na primeira vez)...")
def obter_transcricao(preferido: str = "") -> MotorTranscricao:
    return _descobrir(MOTORES_STT, ORDEM_STT, preferido or os.getenv("VOICE_STT", ""), "transcrição")


@st.cache_resource(show_spinner=False)
def obter_voz(preferido: str = "") -> MotorVoz:
    return _descobrir(MOTORES_TTS, ORDEM_TTS, preferido or os.getenv("VOICE_TTS", ""), "voz")


# ---------- Texto para fala ----------
_SEPARADOR_TABELA = re.compile(r"^\|?\s*:?-{2,}")


def _linha_tabela(linha: str) -> list[str]:
    return [c.strip() for c in linha.strip().strip("|").split("|")]


def texto_para_fala(texto: str, limite: int = TTS_MAX_CARACTERES) -> str:
    """Markdown da resposta -> texto natural para ser falado (tabelas viram frases)."""
    frases: list[str] = []
    linhas = re.sub(r"```.*?```", " ", texto or "", flags=re.S).splitlines()
    i = 0
    while i < len(linhas):
        linha = linhas[i].strip()
        if linha.startswith("|"):
            bloco = []
            while i < len(linhas) and linhas[i].strip().startswith("|"):
                bloco.append(linhas[i].strip())
                i += 1
            # 1ª linha é o cabeçalho; ignora o separador |---|
            dados = [_linha_tabela(b) for b in bloco[1:] if not _SEPARADOR_TABELA.match(b)]
            frases += [f"{c[0]}: {', '.join(c[1:])}" if len(c) > 1 else c[0] for c in dados if any(c)]
            continue
        linha = re.sub(r"^#{1,6}\s*|^>\s*|^[-*+]\s+|^\d+[.)]\s+", "", linha)
        if linha:
            frases.append(linha)
        i += 1

    falado = ". ".join(f.rstrip(".;:") for f in frases)
    falado = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", falado)  # links
    falado = re.sub(r"[*_`#]", "", falado)
    falado = re.sub(r"\s+", " ", falado).strip()
    if falado and falado[-1] not in ".!?":
        falado += "."
    if len(falado) > limite:
        corte = falado.rfind(". ", 0, limite)
        falado = falado[: corte + 1 if corte > 0 else limite].rstrip() + " Os demais detalhes estão na tela."
    return falado


# Frases que o Whisper costuma "inventar" em silêncio ou ruído (comuns em português)
_ALUCINACOES = ("legendas pela comunidade amara.org", "legenda adriana zanotto", "inscreva-se no canal",
                "obrigado por assistir", "obrigado.", "obrigada.", "tchau.", "...")
_ENCERRAR = ("encerrar conversa", "encerrar a conversa", "parar de ouvir", "desligar o microfone",
             "fim da conversa", "pode parar")


def _simples(texto: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t).strip()


def eh_ruido(texto: str) -> bool:
    """Transcrição vazia, curta demais ou alucinação típica do Whisper: ignora e volta a ouvir."""
    t = _simples(texto)
    return len(re.sub(r"[^a-z0-9]", "", t)) < 3 or t in _ALUCINACOES or any(a in t for a in _ALUCINACOES[:4])


def pede_encerrar(texto: str) -> bool:
    t = _simples(texto)
    return any(frase in t for frase in _ENCERRAR)


def responder_em_audio(texto: str) -> Audio:
    return obter_voz().sintetizar(texto_para_fala(texto))


# ---------- Autoteste: python voice.py ----------
def autoteste() -> int:
    """Ida e volta real: sintetiza uma frase, transcreve o áudio e compara. Precisa de internet
    para a síntese e, na primeira vez, para baixar o modelo do Whisper."""
    import tempfile

    frase = "Qual foi a receita de esportes em novembro?"
    print(f"Voz: {TTS_VOZ} | modelo de transcrição: {STT_MODELO}")
    t0 = time.perf_counter()
    try:
        audio = obter_voz().sintetizar(frase)
    except Exception as e:
        print(f"[1/2] FALHA na síntese ({time.perf_counter() - t0:.1f}s): {type(e).__name__}: {e}")
        return 1
    print(f"[1/2] Síntese OK ({time.perf_counter() - t0:.1f}s, {len(audio.dados) / 1024:.0f} KB)")
    destino = Path(tempfile.gettempdir()) / "data_pilot_teste_voz.mp3"
    destino.write_bytes(audio.dados)
    print(f"      Áudio salvo em {destino} (abra para ouvir)")

    t0 = time.perf_counter()
    try:
        texto = obter_transcricao().transcrever(audio.dados, ["Esportes", "Eletrônicos"])
    except Exception as e:
        print(f"[2/2] FALHA na transcrição ({time.perf_counter() - t0:.1f}s): {type(e).__name__}: {e}")
        return 1
    print(f"[2/2] Transcrição OK ({time.perf_counter() - t0:.1f}s, inclui carregar o modelo): {texto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(autoteste())
