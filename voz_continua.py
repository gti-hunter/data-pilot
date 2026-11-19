"""Conversa contínua por voz: um clique liga o microfone e a conversa segue sem novos cliques.

Componente Streamlit (components v2, roda na própria página, sem iframe). No navegador:
1. ouve o microfone e detecta início e fim da fala pelo volume (como o record_until_silence do OpenJarvis);
2. ao fim da fala, envia o WAV 16 kHz para o Python (trigger "fala");
3. pausa enquanto o Python transcreve, consulta o agente e gera a voz;
4. toca a resposta recebida em `data` e volta a ouvir sozinho.
O microfone fica mudo enquanto a resposta toca, para o agente não ouvir a si mesmo.
"""
from __future__ import annotations

import streamlit as st

_CSS = """
.dpv { font-family: var(--st-font, sans-serif); color: var(--st-text-color, #31333f); }
.dpv button {
  width: 100%; padding: .55rem .75rem; border-radius: .5rem; cursor: pointer; font: inherit;
  border: 1px solid var(--st-border-color, rgba(49,51,63,.2));
  background: var(--st-secondary-background-color, #f0f2f6); color: inherit;
}
.dpv button.ativo { background: var(--st-primary-color, #ff4b4b); border-color: transparent; color: #fff; }
.dpv .status { display: flex; align-items: center; gap: .45rem; margin: .5rem 0 .35rem; font-size: .85rem; }
.dpv .ponto { width: .6rem; height: .6rem; border-radius: 50%; background: #9aa0a6; flex: none; }
.dpv[data-estado="ouvindo"] .ponto { background: #21c354; animation: pulso 1.2s infinite; }
.dpv[data-estado="gravando"] .ponto { background: var(--st-primary-color, #ff4b4b); }
.dpv[data-estado="processando"] .ponto { background: #faca2b; }
.dpv[data-estado="falando"] .ponto { background: #1c83e1; }
.dpv[data-estado="erro"] .ponto { background: var(--st-primary-color, #ff4b4b); }
.dpv .nivel {
  height: .3rem; border-radius: .2rem; overflow: hidden;
  background: var(--st-secondary-background-color, #f0f2f6);
}
.dpv .nivel > div { height: 100%; width: 0; background: var(--st-primary-color, #ff4b4b); transition: width .08s; }
.dpv .dica { font-size: .75rem; opacity: .7; margin-top: .35rem; }
@keyframes pulso { 50% { opacity: .35; } }
"""

_JS = r"""
const WORKLET = `
class Coletor extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(2048); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice(0)); this.n = 0; }
    }
    return true;
  }
}
registerProcessor('dp-coletor', Coletor);`;

// Parâmetros de detecção de fala (mesma ideia do record_until_silence do OpenJarvis)
const SILENCIO_FIM_S = 1.2;      // silêncio após a fala que encerra a pergunta
const FALA_MINIMA_S = 0.4;       // descarta estalos e ruídos curtos
const FALA_MAXIMA_S = 30;        // limite de segurança
const PRE_ROLL_S = 0.4;          // guarda o início da fala antes do disparo
const LIMIAR_MINIMO = 0.012;     // RMS mínimo considerado voz
const ESPERA_RESPOSTA_S = 120;   // se o Python não responder, volta a ouvir

const TEXTOS = {
  parado: 'Clique para conversar por voz',
  ouvindo: 'Ouvindo... faça sua pergunta',
  gravando: 'Gravando...',
  processando: 'Pensando na resposta...',
  falando: 'Respondendo...',
};

function paraWav16k(amostras, taxa) {
  const alvo = 16000, razao = taxa / alvo, n = Math.floor(amostras.length / razao);
  const buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const txt = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  txt(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); txt(8, 'WAVE'); txt(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, alvo, true); v.setUint32(28, alvo * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  txt(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {  // média do intervalo: reamostragem com filtro simples
    const ini = Math.floor(i * razao), fim = Math.min(amostras.length, Math.floor((i + 1) * razao));
    let s = 0; for (let j = ini; j < fim; j++) s += amostras[j];
    s /= Math.max(1, fim - ini);
    v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true);
  }
  return buf;
}

function base64(buf) {
  const b = new Uint8Array(buf); let bin = '';
  for (let i = 0; i < b.length; i += 0x8000) bin += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
  return btoa(bin);
}

function mostrar(g, estado, texto) {
  g.estado = estado;
  const ui = g.ui; if (!ui) return;
  ui.raiz.dataset.estado = estado;
  ui.texto.textContent = texto || TEXTOS[estado] || '';
  const ligado = g.ativo;
  ui.botao.textContent = ligado ? 'Encerrar conversa por voz' : 'Iniciar conversa por voz';
  ui.botao.classList.toggle('ativo', ligado);
  if (estado !== 'ouvindo' && estado !== 'gravando') ui.nivel.style.width = '0%';
}

function ouvir(g) {
  if (!g.ativo) return;
  clearTimeout(g.timerEspera);
  g.emFala = false; g.trechos = []; g.preRoll = []; g.silencio = 0; g.duracao = 0; g.acima = 0;
  mostrar(g, 'ouvindo');
}

function enviar(g) {
  const total = g.trechos.reduce((s, t) => s + t.length, 0);
  const juntas = new Float32Array(total); let o = 0;
  for (const t of g.trechos) { juntas.set(t, o); o += t.length; }
  g.trechos = [];
  g.seq += 1;
  mostrar(g, 'processando');
  g.setTriggerValue('fala', { id: `${Date.now()}-${g.seq}`, wav: base64(paraWav16k(juntas, g.ctx.sampleRate)) });
  g.timerEspera = setTimeout(() => ouvir(g), ESPERA_RESPOSTA_S * 1000);
}

function processarTrecho(g, x) {
  let soma = 0; for (let i = 0; i < x.length; i++) soma += x[i] * x[i];
  const rms = Math.sqrt(soma / x.length), dur = x.length / g.ctx.sampleRate;
  if (g.ui && (g.estado === 'ouvindo' || g.estado === 'gravando'))
    g.ui.nivel.style.width = Math.min(100, rms * 900) + '%';
  if (g.estado !== 'ouvindo' && g.estado !== 'gravando') return;  // pausado: processando ou falando

  const limiar = Math.max(LIMIAR_MINIMO, g.ruido * 3);
  if (!g.emFala) {
    g.ruido = g.ruido * 0.95 + rms * 0.05;  // nível de ruído do ambiente
    g.preRoll.push(x); while (g.preRoll.length * dur > PRE_ROLL_S) g.preRoll.shift();
    g.acima = rms > limiar ? g.acima + 1 : 0;
    if (g.acima >= 2) {  // ~90 ms acima do limiar: começou a falar
      g.emFala = true; g.trechos = g.preRoll.slice(); g.preRoll = []; g.duracao = 0; g.silencio = 0;
      mostrar(g, 'gravando');
    }
    return;
  }
  g.trechos.push(x); g.duracao += dur;
  g.silencio = rms < limiar * 0.7 ? g.silencio + dur : 0;
  if (g.silencio >= SILENCIO_FIM_S || g.duracao >= FALA_MAXIMA_S) {
    if (g.duracao - g.silencio < FALA_MINIMA_S) { ouvir(g); return; }
    enviar(g);
  }
}

async function iniciar(g) {
  try {
    g.stream = await navigator.mediaDevices.getUserMedia(
      { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    g.ctx = new AudioContext();
    await g.ctx.resume();
    const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
    await g.ctx.audioWorklet.addModule(url);
    const fonte = g.ctx.createMediaStreamSource(g.stream);
    g.no = new AudioWorkletNode(g.ctx, 'dp-coletor');
    g.no.port.onmessage = (e) => processarTrecho(g, e.data);
    const mudo = g.ctx.createGain(); mudo.gain.value = 0;  // mantém o grafo ativo sem eco no alto-falante
    fonte.connect(g.no).connect(mudo).connect(g.ctx.destination);
    g.ruido = 0.005; g.ativo = true;
    ouvir(g);
  } catch (e) {
    parar(g);
    mostrar(g, 'erro', 'Microfone indisponível: permita o acesso no navegador (' + (e.name || e) + ').');
  }
}

function parar(g, texto) {
  g.ativo = false;
  clearTimeout(g.timerEspera);
  if (g.player) { g.player.pause(); g.player = null; }
  if (g.no) { g.no.port.onmessage = null; g.no.disconnect(); g.no = null; }
  if (g.stream) { g.stream.getTracks().forEach((t) => t.stop()); g.stream = null; }
  if (g.ctx) { g.ctx.close().catch(() => {}); g.ctx = null; }
  mostrar(g, 'parado', texto);
}

function tocar(g, audio) {
  mostrar(g, 'falando');
  const player = new Audio(`data:${audio.mime || 'audio/mpeg'};base64,${audio.b64}`);
  g.player = player;
  const seguir = () => { if (g.player === player) { g.player = null; setTimeout(() => ouvir(g), 300); } };
  player.onended = seguir; player.onerror = seguir;
  player.play().catch(seguir);
}

export default function (component) {
  const { data, setTriggerValue, parentElement } = component;
  const d = data || {};
  const g = (window.__dataPilotVoz ||= { estado: 'parado', ativo: false, seq: 0, ultimoTurno: d.turno ?? 0 });
  g.setTriggerValue = setTriggerValue;

  if (!parentElement.querySelector('.dpv')) {
    const raiz = document.createElement('div');
    raiz.className = 'dpv';
    raiz.innerHTML = '<button type="button"></button><div class="status"><span class="ponto"></span>' +
      '<span class="texto"></span></div><div class="nivel"><div></div></div>' +
      '<div class="dica">Fale e faça uma pausa: a resposta é falada e o microfone volta a ouvir. ' +
      'Diga "encerrar conversa" para parar.</div>';
    parentElement.appendChild(raiz);
    const botao = raiz.querySelector('button');
    botao.addEventListener('click', () => (g.ativo ? parar(g, 'Conversa por voz encerrada') : iniciar(g)));
    g.ui = { raiz, botao, texto: raiz.querySelector('.texto'), nivel: raiz.querySelector('.nivel > div') };
  }
  mostrar(g, g.estado);

  if ((d.turno ?? 0) > g.ultimoTurno) {  // resposta do Python para a última fala
    g.ultimoTurno = d.turno;
    clearTimeout(g.timerEspera);
    if (d.parar) parar(g, 'Conversa por voz encerrada');
    else if (d.audio && g.ativo) tocar(g, d.audio);
    else ouvir(g);
  }
  return () => parar(g);  // só roda quando o componente sai da tela (ex.: mudar para o Painel)
}
"""

_componente = st.components.v2.component("data_pilot.voz_continua", css=_CSS, js=_JS)


def conversa_por_voz(turno: int, audio: dict | None = None, parar: bool = False,
                     key: str = "voz_continua") -> dict | None:
    """Monta o controle de conversa contínua. `turno` sobe a cada resposta do Python; `audio`
    ({"b64", "mime"}) é tocado no navegador. Retorna a fala capturada ({"id", "wav"}) ou None."""
    resultado = _componente(data={"turno": turno, "audio": audio, "parar": parar}, key=key,
                            on_fala_change=lambda: None)
    return getattr(resultado, "fala", None) if resultado is not None else None
