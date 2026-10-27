"""Acesso ao OpenRouter: catálogo de modelos gratuitos e chamadas com registro de evidências.

Decisões de implementação (detalhadas no notebook):
- A chave é lida só da variável de ambiente OPENROUTER_API_KEY e nunca é exibida nem salva.
- O catálogo (/models) é público: a chave NÃO é enviada nessa chamada.
- Erros transitórios (429, 5xx, timeout) são repetidos com espera exponencial; erros
  permanentes (400, 401, 402, 403, 404) não, para não gastar a cota diária gratuita.
- Cada chamada gera um registro com metadados verificáveis (id da geração, provedor,
  tokens, latência, horário UTC, tentativas), que é a evidência de execução.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

from .seguranca import NOME_VARIAVEL_CHAVE, mascarar

def _base_url() -> str:
    """Endpoint da API. A troca via OPENROUTER_BASE_URL existe só para testes com servidor local;
    qualquer outro destino precisa ser HTTPS, para a chave nunca trafegar sem criptografia."""
    url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    if not url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
        raise ValueError("OPENROUTER_BASE_URL precisa usar https:// (ou apontar para um servidor local de teste).")
    return url


BASE_URL = _base_url()

CABECALHOS_APP = {
    # Identificação opcional do aplicativo no OpenRouter (sem dados pessoais).
    "X-Title": "Atividade - comparacao de modelos gratuitos",
}

STATUS_TRANSITORIOS = {408, 429, 500, 502, 503, 504, 520, 522, 524, 529}
ESPERA_MAXIMA_S = 90  # acima disso o limite é tratado como diário/longo e a chamada é abandonada

PALAVRAS_EXCLUIDAS = ("safety", "guard", "moderat", "embed", "rerank", "router", "tts", "whisper")


def agora_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- chave

def _carregar_dotenv() -> None:
    """Carrega o arquivo .env da pasta do notebook, aceitando UTF-8 (com ou sem BOM), UTF-16 e ANSI."""
    caminho = Path(".env")
    if not caminho.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Há um arquivo .env, mas o pacote python-dotenv não está instalado "
                           "(rode: pip install -r requirements.txt).") from exc
    dados = caminho.read_bytes()
    if dados.startswith((b"\xff\xfe", b"\xfe\xff")):  # UTF-16, ex.: "echo ... > .env" no PowerShell
        texto = dados.decode("utf-16")
    else:
        try:
            texto = dados.decode("utf-8-sig")  # UTF-8, com ou sem BOM (Bloco de Notas, VS Code)
        except UnicodeDecodeError:
            texto = dados.decode("cp1252", errors="replace")  # ANSI de editores antigos
    load_dotenv(stream=io.StringIO(texto), override=False)


def obter_chave(permitir_dotenv: bool = True) -> str:
    """Lê OPENROUTER_API_KEY do ambiente ou do arquivo .env na pasta do notebook (fora da entrega)."""
    if permitir_dotenv and not os.environ.get(NOME_VARIAVEL_CHAVE):
        _carregar_dotenv()
    chave = os.environ.get(NOME_VARIAVEL_CHAVE, "").strip()
    if not chave:
        if Path(".env.txt").exists():
            dica = "Há um arquivo .env.txt na pasta: renomeie-o para .env, sem o .txt."
        elif Path(".env").exists():
            dica = "O arquivo .env existe, mas falta a linha OPENROUTER_API_KEY=<sua-chave>."
        else:
            dica = "Defina-a no terminal antes de abrir o Jupyter ou crie o arquivo .env (veja o README)."
        raise RuntimeError(f"A chave {NOME_VARIAVEL_CHAVE} não foi encontrada. {dica}")
    return chave


def _cabecalhos(chave: str) -> dict:
    return {"Authorization": f"Bearer {chave}", "Content-Type": "application/json", **CABECALHOS_APP}


# --------------------------------------------------------------------------- catálogo

def listar_modelos(timeout: int = 30) -> list[dict]:
    """Baixa o catálogo público de modelos (sem enviar a chave)."""
    resposta = requests.get(f"{BASE_URL}/models", timeout=timeout, headers=CABECALHOS_APP)
    resposta.raise_for_status()
    return resposta.json().get("data", [])


def _preco(valor) -> float | None:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def eh_gratuito(modelo: dict) -> bool:
    """Gratuito = id marcado com ':free' ou todos os preços do catálogo iguais a zero."""
    if modelo.get("id", "").endswith(":free"):
        return True
    # Sem a marca ':free', só conta como gratuito se TODOS os preços forem zero: há modelos com
    # entrada e saída zeradas que cobram por requisição, imagem ou áudio.
    precos = [_preco(v) for v in (modelo.get("pricing") or {}).values() if not isinstance(v, (dict, list))]
    return bool(precos) and all(p == 0 for p in precos)


def _modalidades(modelo: dict) -> tuple[list[str], list[str]]:
    arq = modelo.get("architecture") or {}
    entrada = arq.get("input_modalities") or []
    saida = arq.get("output_modalities") or []
    if not entrada and not saida and "modality" in arq:  # formato antigo: "text->text"
        partes = str(arq["modality"]).split("->")
        entrada = partes[0].split("+") if partes else []
        saida = partes[1].split("+") if len(partes) > 1 else []
    return entrada, saida


def filtrar_gratuitos(modelos: list[dict]) -> list[dict]:
    """Mantém modelos gratuitos que recebem texto e respondem só com texto, em formato resumido."""
    gratuitos = []
    for m in modelos:
        if not eh_gratuito(m):
            continue
        entrada, saida = _modalidades(m)
        if "text" not in entrada or set(saida) != {"text"}:  # exclui geradores de áudio/imagem
            continue
        criado = m.get("created")
        gratuitos.append({
            "id": m.get("id"),
            "nome": m.get("name"),
            "contexto": m.get("context_length"),
            "entrada": "+".join(entrada),
            "saida": "+".join(saida),
            "parametros_suportados": sorted(m.get("supported_parameters") or []),
            "preco": m.get("pricing"),
            "criado_em": (datetime.fromtimestamp(criado, timezone.utc).date().isoformat()
                          if isinstance(criado, (int, float)) else None),
            "descricao": (m.get("description") or "")[:280],
        })
    return sorted(gratuitos, key=lambda x: x["id"])


def salvar_catalogo(gratuitos: list[dict], total_catalogo: int, pasta: str | Path) -> Path:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / "catalogo_gratuitos.json"
    caminho.write_text(json.dumps({
        "consultado_em_utc": agora_utc(),
        "endpoint": f"{BASE_URL}/models",
        "total_modelos_no_catalogo": total_catalogo,
        "total_gratuitos_texto": len(gratuitos),
        "modelos": gratuitos,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return caminho


def _fornecedor(model_id: str) -> str:
    return model_id.split("/")[0] if "/" in model_id else model_id


def _apto(m: dict) -> bool:
    texto = f"{m['id']} {m.get('nome') or ''}".lower()
    return not m["id"].startswith("openrouter/") and not any(p in texto for p in PALAVRAS_EXCLUIDAS)


def _especializado(m: dict) -> bool:
    """Modelos feitos para agentes de programação (pela descrição do catálogo) ficam no fim da fila,
    porque o prompt da atividade é de uso geral."""
    descricao = (m.get("descricao") or "").lower()
    return any(p in descricao for p in ("coding agent", "coding model"))


def escolher_modelos(gratuitos: list[dict], preferidos: list[str] | None = None,
                     n: int = 2) -> dict:
    """Escolhe n modelos gratuitos de fornecedores diferentes.

    Ordem: (1) preferidos que estejam gratuitos agora; (2) demais modelos aptos, priorizando
    os de uso geral, depois os que aceitam temperature e max_tokens e depois o maior contexto.
    Devolve {'escolhidos', 'reservas', 'justificativas'}. As reservas, usadas para substituição,
    começam pelos preferidos que sobraram.
    """
    por_id = {m["id"]: m for m in gratuitos}
    justificativas, escolhidos, reservas = [], [], []

    for mid in preferidos or []:
        if mid not in por_id:
            justificativas.append(f"{mid}: preferido, mas não está gratuito/disponível agora, foi ignorado.")
        elif len(escolhidos) < n:
            escolhidos.append(mid)
            justificativas.append(f"{mid}: preferido e gratuito no catálogo consultado.")
        else:
            reservas.append(mid)
            justificativas.append(f"{mid}: preferido, fica na lista de reservas.")

    def chave_ordem(m):
        params = set(m["parametros_suportados"])
        return (not _especializado(m), ("temperature" in params) + ("max_tokens" in params),
                m.get("contexto") or 0)

    candidatos = sorted((m for m in gratuitos if _apto(m)), key=chave_ordem, reverse=True)
    for m in candidatos:
        if m["id"] in escolhidos or m["id"] in reservas:
            continue
        if len(escolhidos) < n and _fornecedor(m["id"]) not in {_fornecedor(e) for e in escolhidos}:
            escolhidos.append(m["id"])
            justificativas.append(
                f"{m['id']}: escolhido automaticamente (fornecedor diferente, contexto "
                f"{m.get('contexto')}, aceita temperature/max_tokens: "
                f"{'temperature' in m['parametros_suportados']}/{'max_tokens' in m['parametros_suportados']})."
            )
        else:
            reservas.append(m["id"])

    if len(escolhidos) < n:
        raise RuntimeError(f"Só há {len(escolhidos)} modelo(s) gratuito(s) apto(s) no catálogo agora.")
    return {"escolhidos": escolhidos, "reservas": reservas, "justificativas": justificativas}


# --------------------------------------------------------------------------- limites

def consultar_limites(chave: str, timeout: int = 30) -> dict:
    """Consulta /key e devolve só campos não sensíveis (o rótulo da chave é descartado)."""
    resposta = requests.get(f"{BASE_URL}/key", headers=_cabecalhos(chave), timeout=timeout)
    if resposta.status_code != 200:
        return {"consultado_em_utc": agora_utc(), "erro_http": resposta.status_code}
    dados = resposta.json().get("data", {})
    return {
        "consultado_em_utc": agora_utc(),
        "is_free_tier": dados.get("is_free_tier"),
        "free_model_daily_requests": dados.get("free_model_daily_requests"),
    }


# --------------------------------------------------------------------------- chamadas

def _classificar_erro(status: int | None, mensagem: str) -> str:
    msg = (mensagem or "").lower()
    if status is None:
        return "tempo_esgotado" if "timeout" in msg else "falha_de_rede"
    if status == 400:
        return "requisicao_invalida"
    if status == 401:
        return "chave_invalida"
    if status == 402:
        return "sem_creditos_ou_saldo_negativo"
    if status == 403:
        # 403 também aparece quando o modelo só é liberado para certos aplicativos
        # (ex.: "only available on agentic harnesses"), o que não tem relação com moderação
        return "bloqueado_por_moderacao" if ("moderat" in msg or "flagged" in msg) else "acesso_restrito"
    if status == 404:
        return "politica_de_dados_bloqueia_modelo" if "data policy" in msg else "modelo_indisponivel"
    if status == 408:
        return "tempo_esgotado"
    if status == 429:
        if "per-day" in msg or "per day" in msg:
            return "limite_diario"
        # "rate-limited upstream": o provedor do modelo está congestionado, não é o limite da conta
        return "provedor_sobrecarregado" if "upstream" in msg else "limite_de_taxa"
    if status >= 500:
        return "erro_do_provedor"
    return f"http_{status}"


def _espera(resposta: requests.Response | None, tentativa: int) -> float:
    """Tempo de espera: Retry-After > X-RateLimit-Reset > backoff exponencial com jitter."""
    if resposta is not None:
        retry_after = resposta.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        reset = resposta.headers.get("X-RateLimit-Reset")
        if reset:
            try:
                reset_s = float(reset) / (1000 if float(reset) > 1e11 else 1)
                return max(0.0, reset_s - time.time())
            except ValueError:
                pass
    return min(60.0, 2.0 * 2 ** tentativa) + random.uniform(0, 1)


def _mensagem_erro(corpo: dict | None, texto_bruto: str) -> str:
    if isinstance(corpo, dict) and isinstance(corpo.get("error"), dict):
        erro = corpo["error"]
        meta = erro.get("metadata") or {}
        extra = meta.get("raw") or meta.get("provider_name") or ""
        return f"{erro.get('message', '')} {extra}".strip()
    return texto_bruto[:300]


def chamar_modelo(chave: str, modelo: str, mensagens: list[dict], parametros: dict,
                  max_tentativas: int = 4, timeout: int = 180, rotulo: str = "") -> dict:
    """Envia uma conversa ao modelo e devolve um registro de evidência (nunca lança exceção)."""
    corpo_req = {"model": modelo, "messages": mensagens, **parametros}
    registro = {
        "execucao_id": uuid.uuid4().hex[:12],
        "rotulo": rotulo,
        "modelo_solicitado": modelo,
        "parametros": parametros,
        "prompt_sha256": sha256(json.dumps(mensagens, ensure_ascii=False, sort_keys=True)),
        "inicio_utc": agora_utc(),
        "tentativas": 0,
        "esperas_s": [],
        "erros_por_tentativa": [],  # erros contornados por nova tentativa também são evidência
        "sucesso": False,
    }
    inicio_total = time.perf_counter()
    resposta = None

    for tentativa in range(max_tentativas):
        registro["tentativas"] = tentativa + 1
        t0 = time.perf_counter()
        try:
            resposta = requests.post(f"{BASE_URL}/chat/completions", headers=_cabecalhos(chave),
                                     json=corpo_req, timeout=timeout)
            status, texto_bruto = resposta.status_code, resposta.text
        except requests.RequestException as exc:
            resposta, status, texto_bruto = None, None, f"{type(exc).__name__}: {exc}"
        latencia = time.perf_counter() - t0

        try:
            corpo = json.loads(texto_bruto) if status is not None else None
        except (json.JSONDecodeError, TypeError):
            corpo = None

        erro_no_corpo = isinstance(corpo, dict) and "error" in corpo
        escolha = (corpo or {}).get("choices", [{}])[0] if isinstance(corpo, dict) and corpo.get("choices") else {}
        erro_na_escolha = isinstance(escolha, dict) and "error" in escolha

        if status == 200 and not erro_no_corpo and not erro_na_escolha and escolha:
            msg = escolha.get("message") or {}
            uso = corpo.get("usage") or {}
            detalhes = uso.get("completion_tokens_details") or {}
            raciocinio = msg.get("reasoning") or ""
            conteudo = msg.get("content") or ""
            if not isinstance(raciocinio, str):
                raciocinio = json.dumps(raciocinio, ensure_ascii=False)
            if not isinstance(conteudo, str):
                conteudo = json.dumps(conteudo, ensure_ascii=False)
            registro.pop("erro_tipo", None)      # o erro de uma tentativa anterior foi contornado
            registro.pop("erro_mensagem", None)
            registro.update({
                "sucesso": True,
                "status_http": status,
                "latencia_s": round(latencia, 3),
                "modelo_respondido": corpo.get("model"),
                "provedor": corpo.get("provider"),
                "geracao_id": corpo.get("id"),
                "finish_reason": escolha.get("finish_reason"),
                "native_finish_reason": escolha.get("native_finish_reason"),
                "resposta": conteudo,
                "raciocinio": raciocinio or None,
                "tokens_prompt": uso.get("prompt_tokens"),
                "tokens_completion": uso.get("completion_tokens"),
                "tokens_raciocinio": detalhes.get("reasoning_tokens"),
                "tokens_total": uso.get("total_tokens"),
                "custo_informado": uso.get("cost"),
                "ratelimit_restante": resposta.headers.get("X-RateLimit-Remaining") if resposta is not None else None,
            })
            if not conteudo.strip():
                registro["alerta"] = ("resposta vazia" + (" (o raciocínio consumiu o limite de tokens)"
                                                          if raciocinio and escolha.get("finish_reason") == "length" else ""))
            break

        # ---- falha: classificar e decidir se vale repetir
        mensagem = mascarar(_mensagem_erro(corpo if erro_no_corpo else (escolha if erro_na_escolha else None),
                                           texto_bruto), [chave])
        status_efetivo = status
        if status == 200:  # erro dentro de uma resposta 200 (ex.: falha do provedor no meio da geração)
            erro = (escolha.get("error") if erro_na_escolha else
                    corpo.get("error") if erro_no_corpo else None) or {}
            status_efetivo = erro.get("code") if isinstance(erro.get("code"), int) else 502
        tipo = _classificar_erro(status_efetivo if isinstance(status_efetivo, int) else None, mensagem)
        registro["erros_por_tentativa"].append({"tentativa": tentativa + 1, "status_http": status,
                                                "erro_tipo": tipo, "erro_mensagem": mensagem})
        registro.update({"status_http": status, "erro_tipo": tipo, "erro_mensagem": mensagem,
                         "latencia_s": round(latencia, 3)})

        transitorio = status is None or status_efetivo in STATUS_TRANSITORIOS
        timeouts = sum(e["erro_tipo"] == "tempo_esgotado" for e in registro["erros_por_tentativa"])
        if not transitorio or tipo == "limite_diario" or tentativa == max_tentativas - 1 or timeouts >= 2:
            break  # no máximo 2 tempos esgotados, para um modelo lento não prender a execução
        espera = _espera(resposta, tentativa)
        if espera > ESPERA_MAXIMA_S:
            registro["erro_tipo"] = "limite_de_taxa_longo"
            registro["erro_mensagem"] += f" (servidor pediu espera de {espera:.0f}s; chamada abandonada)"
            break
        registro["esperas_s"].append(round(espera, 1))
        time.sleep(espera)

    registro["fim_utc"] = agora_utc()
    registro["duracao_total_s"] = round(time.perf_counter() - inicio_total, 3)
    return registro


def salvar_registro(registro: dict, caminho: str | Path) -> None:
    """Acrescenta o registro em JSONL (evidência preservada mesmo se a execução parar no meio)."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    texto = mascarar(json.dumps(registro, ensure_ascii=False))
    with caminho.open("a", encoding="utf-8") as f:
        f.write(texto + "\n")


# Erros que, na 1ª rodada, levam à troca do modelo por uma reserva. Provedor sobrecarregado e
# erro do provedor só entram aqui depois de esgotadas as novas tentativas.
ERROS_DE_INDISPONIBILIDADE = {"modelo_indisponivel", "politica_de_dados_bloqueia_modelo", "requisicao_invalida",
                              "acesso_restrito", "provedor_sobrecarregado", "erro_do_provedor", "tempo_esgotado"}
MOTIVOS_DE_INTERRUPCAO = {
    "limite_diario": "limite diário de requisições gratuitas atingido",
    "limite_de_taxa_longo": "o servidor pediu uma espera longa demais",
    "chave_invalida": "chave inválida ou revogada (erro 401)",
    "sem_creditos_ou_saldo_negativo": "saldo da conta negativo (erro 402)",
}
ERROS_QUE_INTERROMPEM = set(MOTIVOS_DE_INTERRUPCAO)
MAX_TROCAS_POR_MODELO = 3


def rodar_comparacao(chave: str, modelos: list[str], mensagens: list[dict], parametros: dict,
                     repeticoes: int, arquivo_jsonl: str | Path, reservas: list[str] | None = None,
                     substituir_se_indisponivel: bool = True, pausa_s: float = 4.0,
                     log=print) -> dict:
    """Executa o mesmo prompt nos modelos, intercalando a ordem a cada rodada.

    Se um modelo falhar por indisponibilidade na primeira rodada, ele é trocado pela próxima
    reserva (a decisão é registrada). Devolve {'registros', 'modelos_finais', 'decisoes'}.
    """
    modelos = list(modelos)
    reservas = [r for r in (reservas or []) if r not in modelos]
    registros, decisoes = [], []
    fornecedores_indisponiveis = set()  # outro modelo do mesmo fornecedor tende a falhar igual

    for rodada in range(1, repeticoes + 1):
        ordem = list(modelos) if rodada % 2 == 1 else list(reversed(modelos))
        for posicao, modelo in enumerate(ordem):
            atual, trocas = modelo, 0
            while True:
                reg = chamar_modelo(chave, atual, mensagens, parametros,
                                    rotulo=f"rodada {rodada}, posição {posicao + 1}")
                reg["rodada"] = rodada
                registros.append(reg)
                salvar_registro(reg, arquivo_jsonl)
                if reg["sucesso"]:
                    contornados = [e["erro_tipo"] for e in reg["erros_por_tentativa"]]
                    log(f"[rodada {rodada}] {atual}: ok em {reg['latencia_s']}s "
                        f"({reg.get('tokens_completion')} tokens, provedor {reg.get('provedor')})"
                        + (f", depois de {len(contornados)} erro(s) contornado(s): {', '.join(contornados)}"
                           if contornados else ""))
                else:
                    log(f"[rodada {rodada}] {atual}: FALHOU ({reg['erro_tipo']}): {reg['erro_mensagem'][:160]}")

                pode_trocar = (substituir_se_indisponivel and rodada == 1 and not reg["sucesso"]
                               and reg.get("erro_tipo") in ERROS_DE_INDISPONIBILIDADE
                               and trocas < MAX_TROCAS_POR_MODELO)
                if pode_trocar:
                    fornecedores_indisponiveis.add(_fornecedor(atual))
                evitar = {_fornecedor(m) for m in modelos if m != atual}
                livres = ([r for r in reservas if _fornecedor(r) not in evitar | fornecedores_indisponiveis]
                          or [r for r in reservas if _fornecedor(r) not in evitar] or list(reservas))
                if pode_trocar and livres:
                    substituto = livres[0]
                    reservas.remove(substituto)
                    modelos[modelos.index(atual)] = substituto
                    decisoes.append({"momento_utc": agora_utc(), "decisao": "substituição de modelo",
                                     "de": atual, "para": substituto,
                                     "motivo": f"{reg['erro_tipo']}: {reg['erro_mensagem'][:200]}"})
                    log(f"   -> trocando {atual} por {substituto} (troca registrada)")
                    atual, trocas = substituto, trocas + 1
                    time.sleep(pausa_s)
                    continue
                if not reg["sucesso"] and reg.get("erro_tipo") in ERROS_QUE_INTERROMPEM:
                    # esses erros valem para a conta toda (o limite dos modelos :free é por conta),
                    # então seguir chamando só gastaria cota sem produzir evidência nova
                    decisoes.append({"momento_utc": agora_utc(), "decisao": "execução interrompida",
                                     "motivo": f"{reg['erro_tipo']}: {reg['erro_mensagem'][:200]}"})
                    log(f"   -> execução interrompida: {MOTIVOS_DE_INTERRUPCAO[reg['erro_tipo']]}")
                    return {"registros": registros, "modelos_finais": modelos, "decisoes": decisoes}
                break
            time.sleep(pausa_s)  # mantém o ritmo bem abaixo de 20 req/min

    return {"registros": registros, "modelos_finais": modelos, "decisoes": decisoes}
