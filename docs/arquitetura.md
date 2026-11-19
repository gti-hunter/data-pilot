# Data Pilot: arquitetura e regras de negócio

## Princípio

A IA não calcula nenhum número. O modelo de linguagem recebe a pergunta e a lista de ferramentas, decide qual função chamar e com quais parâmetros, e só explica o resultado que o Python devolve. Isso elimina número inventado e permite usar um modelo gratuito.

## Camadas

### 1. Dados (`data.py`)

- Lê o `sales.csv`, converte `data` para data, padroniza `status` (minúsculas, sem espaços), remove espaços extras de `cliente`, `cidade` e `categoria` e remove linhas duplicadas.
- Cria `receita = quantidade × preco_unitario` quando a coluna não existe.
- Salva `sales_clean.csv` com `;` como separador e `,` como decimal, e lê do mesmo jeito.
- A leitura usa `st.cache_data` com a data de modificação do arquivo na chave do cache: se o CSV limpo for regerado, o cache é invalidado. Se o arquivo limpo não existir, ele é gerado a partir do cru.

### 2. Métricas (`metrics.py`)

- Uma função por pergunta de negócio. Todas recebem o DataFrame e os filtros opcionais `categoria`, `data_inicio` e `data_fim`.
- `aplicar_filtros` concentra a regra de filtro: a categoria é comparada sem diferenciar maiúsculas, acentos e plural simples ("esporte" encontra "Esportes") e o período é inclusivo nas duas pontas.
- Se a regra de uma métrica mudar, ela muda em um lugar só e vale para o painel e para o chat.

### 3. Painel (`app.py`)

- Filtros de categoria e período, 6 cards e 2 gráficos Plotly (receita mensal e top 5 cidades).
- Formatação no padrão brasileiro: valores com `fmt_brl`/`fmt_num` e gráficos com `separators=",."`.
- O app só roda com `streamlit run app.py`; executado como script comum, ele avisa e encerra.

### 4. Agente (`agent.py`)

Fluxo de uma pergunta:

1. O usuário pergunta: "Qual a receita por categoria no 1º trimestre?"
2. O app envia ao modelo as instruções (com a data de hoje, o período dos dados e as categorias), o histórico e a lista `TOOLS`.
3. O modelo responde pedindo `obter_resumo_por(dimensao="categoria", data_inicio="2026-01-01", data_fim="2026-03-31")`.
4. `_executar_tool` confere se o nome está entre as ferramentas declaradas, executa a função de `metrics.py` e devolve o resultado como JSON.
5. O modelo recebe os números e escreve a resposta final em português.

Proteções:

- Só funções declaradas em `TOOLS` podem ser executadas; argumentos vazios ou inválidos viram mensagem de erro para o modelo, sem derrubar o chat.
- No máximo 4 rodadas por pergunta; na última o modelo é obrigado a responder sem novas ferramentas.
- Limite de tempo por chamada (`LLM_TIMEOUT`, padrão 20 s), sem novas tentativas automáticas do SDK; erros de limite de uso, chave inválida e conexão viram mensagens claras com o motivo real do provedor.
- Sobrecarga do modelo (HTTP 5xx como "high demand", 429 ou tempo esgotado): a pergunta é refeita do zero no próximo candidato, porque as assinaturas de raciocínio de um modelo não valem em outro. Ordem: o que funcionou por último, o modelo configurado, os alternativos do mesmo provedor (no Gemini, os Flash estáveis liberados para a chave, Lite primeiro) e o outro provedor, se houver chave. Limites: 4 tentativas e `LLM_ORCAMENTO` (60 s). O que funcionou é usado nas perguntas seguintes.
- A mensagem do modelo com as chamadas de ferramenta é reenviada intacta na rodada seguinte, como exige o Gemini 3.

Provedores (API compatível com OpenAI, mesmo SDK):

| Provedor | Base URL | Modelo padrão |
|---|---|---|
| OpenRouter (padrão) | `https://openrouter.ai/api/v1` | `openrouter/free` |
| Google Gemini (opcional) | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-3.8-flash` |

Com `LLM_PROVIDER` vazio, o app usa o primeiro provedor com chave configurada, nessa ordem.

No Gemini, com `GEMINI_AUTH=auto`, o agente usa o endpoint compatível com OpenAI com `Authorization: Bearer`, que exige esse header (sem ele: "Missing or invalid Authorization header") e aceita chaves `AIza` e `AQ.` ("Auth key", padrão do Google AI Studio desde 28/05/2026). Se a chave for recusada, tenta a API nativa (`generateContent`) com `x-goog-api-key`, convertendo mensagens e ferramentas para o formato nativo e de volta e preservando a `thoughtSignature` do Gemini 3. O modo que funcionar fica memorizado. O raciocínio só é enviado a modelos 2.5 ou mais novos e é retirado se o modelo o recusar. Os erros chegam com a mensagem real do Google. A temperatura fica no padrão, como o Google recomenda para o Gemini 3. `python agent.py` executa um diagnóstico de conexão, chave, modelo e de uma pergunta real.

### 5. Voz (`voice.py`)

Camada opcional de entrada e saída, no mesmo desenho do módulo `speech/` do OpenJarvis: interfaces de motor de transcrição e de voz, registro de motores e descoberta automática (tenta o preferido e cai para o próximo se um não estiver instalado).

| Etapa | Motor | Onde roda |
|---|---|---|
| Gravação | `st.chat_input(accept_audio=True)` | Navegador (WAV 16 kHz) |
| Transcrição | faster-whisper `base`, CPU int8, idioma pt, filtro de silêncio, vocabulário do negócio no `initial_prompt` | Local |
| Resposta | Mesmo `executar_agente` (tool calling) | — |
| Fala | edge-tts, `pt-BR-FranciscaNeural`, MP3 | Serviço do Microsoft Edge (só o texto) |

- `texto_para_fala` adapta a resposta: remove markdown, transforma linhas de tabela em frases ("Eletrônicos: R$ 411.698,05") e limita o tamanho (`VOICE_TTS_MAX_CHARS`).
- Os motores são carregados uma vez por processo (`st.cache_resource`). Falha na voz não bloqueia a resposta em texto.
- **Conversa contínua (`voz_continua.py`)**: componente Streamlit v2 (roda na página, sem iframe). Um clique liga o microfone; um AudioWorklet mede o volume, um limiar adaptativo ao ruído detecta o início da fala (com 0,4 s de pré-gravação) e 1,2 s de silêncio encerra a pergunta, como o `record_until_silence` do OpenJarvis. O WAV 16 kHz vai para o Python como evento; o navegador pausa a escuta, o Python transcreve, consulta o agente, gera a voz e devolve o áudio no próximo turno; o navegador toca e volta a ouvir. Falas curtas demais, ruído e alucinações típicas do Whisper são descartados; "encerrar conversa" desliga o microfone. Sair da aba também desliga.
- Diferenças em relação ao OpenJarvis: o Kokoro (voz local) exige Python < 3.13 e o projeto usa 3.14, por isso a voz usa edge-tts; a gravação acontece no navegador, e não pelo `sounddevice` do servidor.

## Regras de negócio

1. **Receita**: `quantidade × preco_unitario`.
2. **Pedidos concluídos**: receita, pedidos, ticket médio, clientes únicos, receita média por cliente, top cidades e receita mensal consideram apenas o status `concluído`.
3. **Taxa de cancelamento**: pedidos `cancelado` divididos pelo total de pedidos de qualquer status, vezes 100.
4. **Receita média por cliente**: receita dos pedidos concluídos dividida pelos clientes únicos com pedido concluído.

## Privacidade

O modelo recebe apenas resultados agregados das funções. Linhas da base e nomes de clientes nunca são enviados ao LLM.
