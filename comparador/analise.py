"""Métricas objetivas sobre as respostas e tabelas comparativas.

O prompt da atividade foi desenhado para ter partes verificáveis automaticamente:
formato (3 seções em ordem), limite de palavras, idioma, resultado numérico do cálculo,
número de itens da lista e número de frases do conceito. São heurísticas simples e
transparentes: servem de evidência, mas não substituem a leitura das respostas.
"""
from __future__ import annotations

import itertools
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

STOPWORDS_PT = {"de", "que", "o", "a", "os", "as", "um", "uma", "para", "com", "não", "nao", "em",
                "do", "da", "dos", "das", "no", "na", "é", "e", "se", "por", "mais", "como", "ao",
                "quando", "seu", "sua", "pode", "são", "sao", "isso", "está", "esta", "modelo", "dados"}
STOPWORDS_EN = {"the", "of", "and", "to", "in", "is", "that", "for", "it", "with", "as", "on", "are",
                "this", "be", "by", "or", "an", "can", "when", "model", "data", "which", "its"}


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return sem_acento.lower().strip()


def palavras(texto: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:[-'][A-Za-zÀ-ÿ0-9]+)*", texto or "")


def _titulo_da_linha(linha: str, titulos: list[str]) -> str | None:
    """Reconhece '### Conceito', '**Conceito**', 'Conceito:', '1. Conceito' etc."""
    limpo = normalizar(re.sub(r"^[\s#>*_\d.)-]+|[\s*_:]+$", "", linha))
    for t in titulos:
        if limpo == normalizar(t):
            return t
    return None


def extrair_secoes(texto: str, titulos: list[str]) -> tuple[dict, list[str]]:
    """Divide a resposta pelas seções. Devolve (conteúdo por título, ordem encontrada)."""
    secoes, ordem, atual = {}, [], None
    for linha in (texto or "").splitlines():
        t = _titulo_da_linha(linha, titulos)
        if t and t not in secoes:
            atual = t
            secoes[t] = []
            ordem.append(t)
        elif atual:
            secoes[atual].append(linha)
    return {k: "\n".join(v).strip() for k, v in secoes.items()}, ordem


def contar_frases(texto: str) -> int:
    sem_numeros = re.sub(r"\d+[.,]\d+", "N", texto or "")
    partes = [p for p in re.split(r"[.!?]+(?:\s|$)", sem_numeros) if len(palavras(p)) >= 3]
    return len(partes)


def contar_itens_lista(texto: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S", texto or ""))


def idioma_provavel(texto: str) -> str:
    toks = [normalizar(p) for p in palavras(texto)]
    pt = sum(t in {normalizar(s) for s in STOPWORDS_PT} for t in toks)
    en = sum(t in STOPWORDS_EN for t in toks)
    if pt == en == 0:
        return "indefinido"
    return "pt" if pt >= 2 * en else ("en" if en >= 2 * pt else "misto")


def contem_valor(texto: str, valor: float) -> bool:
    """Procura o valor monetário (ex.: 54 em 'R$ 54,00', '54.00' ou '54 reais')."""
    inteiro = int(valor)
    padrao = rf"(?<![\d.,]){inteiro}(?:[.,]00?)?(?![\d])"
    return re.search(padrao, texto or "") is not None


def avaliar_resposta(texto: str, tarefa: dict) -> dict:
    """Aplica as verificações objetivas definidas em `tarefa` (ver notebook)."""
    titulos = tarefa["titulos"]
    secoes, ordem = extrair_secoes(texto, titulos)
    n_palavras = len(palavras(texto))
    texto_calculo = secoes.get(tarefa["secao_calculo"], texto)
    texto_lista = secoes.get(tarefa["secao_lista"], "")
    texto_conceito = secoes.get(tarefa["secao_conceito"], "")

    checks = {
        "formato_ok": ordem == titulos,
        "limite_palavras_ok": 0 < n_palavras <= tarefa["max_palavras"],
        "idioma_pt_ok": idioma_provavel(texto) == "pt",
        "calculo_ok": contem_valor(texto_calculo, tarefa["resposta_calculo"]),
        "lista_ok": contar_itens_lista(texto_lista) == tarefa["itens_lista"],
        "conceito_frases_ok": 0 < contar_frases(texto_conceito) <= tarefa["max_frases_conceito"],
    }
    return {
        **checks,
        "pontuacao_objetiva": round(sum(checks.values()) / len(checks), 3),
        "palavras": n_palavras,
        "secoes_encontradas": " > ".join(ordem) or "(nenhuma)",
        "itens_na_lista": contar_itens_lista(texto_lista),
        "frases_no_conceito": contar_frases(texto_conceito),
        "idioma": idioma_provavel(texto),
    }


CRITERIOS = ["formato_ok", "limite_palavras_ok", "idioma_pt_ok", "calculo_ok", "lista_ok", "conceito_frases_ok"]


def tabela_execucoes(registros: list[dict], tarefa: dict) -> pd.DataFrame:
    """Uma linha por chamada, com metadados de execução e métricas da resposta."""
    linhas = []
    for r in registros:
        base = {
            "rodada": r.get("rodada"),
            "modelo": r["modelo_solicitado"],
            "inicio_utc": r.get("inicio_utc"),
            "sucesso": r["sucesso"],
            "status_http": r.get("status_http"),
            "tentativas": r.get("tentativas"),
            "latencia_s": r.get("latencia_s") if r["sucesso"] else None,
            "provedor": r.get("provedor"),
            "finish_reason": r.get("finish_reason"),
            "tokens_prompt": r.get("tokens_prompt"),
            "tokens_completion": r.get("tokens_completion"),
            "tokens_raciocinio": r.get("tokens_raciocinio"),
            "geracao_id": r.get("geracao_id"),
            "erro_tipo": r.get("erro_tipo"),
            "erros_contornados": (", ".join(e["erro_tipo"] for e in r.get("erros_por_tentativa", []))
                                  if r["sucesso"] else None) or None,
        }
        if r["sucesso"]:
            # Resposta vazia (ex.: o raciocínio gastou todo o max_tokens) é avaliada como texto vazio,
            # então não cumpre nenhuma regra: o modelo respondeu à chamada, mas não entregou a resposta.
            base["resposta_vazia"] = not (r.get("resposta") or "").strip()
            base.update(avaliar_resposta(r.get("resposta") or "", tarefa))
        linhas.append(base)
    df = pd.DataFrame(linhas)
    for coluna in ("rodada", "status_http", "tentativas", "tokens_prompt", "tokens_completion",
                   "tokens_raciocinio", "palavras", "itens_na_lista", "frases_no_conceito"):
        if coluna in df:  # inteiros que aceitam ausência (falhas não têm tokens)
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce").round().astype("Int64")
    return df


def consistencia(textos: list[str]) -> float | None:
    """Similaridade média entre pares de respostas (1 = idênticas). None se < 2 respostas."""
    pares = list(itertools.combinations([t for t in textos if t], 2))
    if not pares:
        return None
    return round(sum(SequenceMatcher(None, a, b).ratio() for a, b in pares) / len(pares), 3)


def _media(serie: pd.Series, casas: int):
    """Média arredondada que tolera colunas vazias ou só com valores ausentes (devolve None)."""
    valores = pd.to_numeric(serie, errors="coerce").dropna().astype(float)
    return round(float(valores.mean()), casas) if len(valores) else None


def _desvio(serie: pd.Series, casas: int):
    valores = pd.to_numeric(serie, errors="coerce").dropna().astype(float)
    return round(float(valores.std()), casas) if len(valores) > 1 else None


def tabela_comparativa(df: pd.DataFrame, registros: list[dict]) -> pd.DataFrame:
    """Agrega por modelo: confiabilidade, desempenho e aderência às instruções."""
    linhas = []
    for modelo, g in df.groupby("modelo", sort=False):
        ok = g[g["sucesso"]]
        coluna = lambda nome: ok[nome] if nome in ok else pd.Series(dtype=float)
        textos = [r.get("resposta") for r in registros
                  if r["modelo_solicitado"] == modelo and r["sucesso"]]
        linha = {
            "modelo": modelo,
            "chamadas": len(g),
            "sucessos": int(g["sucesso"].sum()),
            "respostas_vazias": int(coluna("resposta_vazia").eq(True).sum()),
            "tentativas_extras": int((g["tentativas"] - 1).clip(lower=0).sum()),
            "provedores": ", ".join(sorted({str(p) for p in ok["provedor"].dropna()})) or "-",
            "latencia_media_s": _media(coluna("latencia_s"), 2),
            "latencia_desvio_s": _desvio(coluna("latencia_s"), 2),
            "tokens_saida_medio": _media(coluna("tokens_completion"), 1),
            "tokens_raciocinio_medio": _media(coluna("tokens_raciocinio"), 1) or 0,
            "palavras_medio": _media(coluna("palavras"), 1),
            "consistencia_entre_rodadas": consistencia(textos),
        }
        for c in CRITERIOS:
            if c in ok and len(ok):
                linha[c] = f"{int(ok[c].eq(True).sum())}/{len(ok)}"
        if "pontuacao_objetiva" in ok and len(ok):
            linha["pontuacao_objetiva_media"] = _media(ok["pontuacao_objetiva"], 3)
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("modelo")


def resumo_automatico(comp: pd.DataFrame) -> str:
    """Resumo curto montado a partir da tabela comparativa (base para a análise escrita)."""
    if len(comp) < 2:
        return "Há menos de dois modelos com dados, então não é possível comparar."
    a, b = comp.index[:2]
    ca, cb = comp.loc[a], comp.loc[b]
    frases = [f"Foram comparados os modelos A ({a}) e B ({b}), com o mesmo prompt e os mesmos parâmetros."]
    frases.append(f"O modelo A respondeu {ca['sucessos']} de {ca['chamadas']} chamadas e o B respondeu "
                  f"{cb['sucessos']} de {cb['chamadas']}, com {ca['tentativas_extras']} e "
                  f"{cb['tentativas_extras']} tentativas extras, respectivamente.")
    vazias_a, vazias_b = int(ca.get("respostas_vazias") or 0), int(cb.get("respostas_vazias") or 0)
    if vazias_a or vazias_b:
        frases.append(f"Mesmo respondendo à chamada, A entregou {vazias_a} e B entregou {vazias_b} resposta(s) vazia(s), "
                      "porque o raciocínio consumiu todo o limite de tokens.")
    if pd.notna(ca.get("latencia_media_s")) and pd.notna(cb.get("latencia_media_s")):
        la, lb = ca["latencia_media_s"], cb["latencia_media_s"]
        rapido = "os dois empataram" if la == lb else f"o modelo {'A' if la < lb else 'B'} foi mais rápido"
        frases.append(f"O tempo médio de resposta foi de {la} s para A e {lb} s para B ({rapido} nesta amostra).")
    if "pontuacao_objetiva_media" in comp.columns:
        pa, pb = ca.get("pontuacao_objetiva_media"), cb.get("pontuacao_objetiva_media")
        if pd.notna(pa) and pd.notna(pb):
            lider = "houve empate" if pa == pb else f"o modelo {'A' if pa > pb else 'B'} se saiu melhor"
            frases.append(f"Nas regras objetivas (escala de 0 a 1), A teve {pa} e B teve {pb}, então {lider}.")
    if "calculo_ok" in comp.columns:
        frases.append(f"O cálculo do troco foi acertado em {ca.get('calculo_ok')} execuções por A "
                      f"e em {cb.get('calculo_ok')} por B.")
    frases.append("Como a amostra é pequena, diferenças pequenas não devem ser tratadas como conclusivas.")
    return " ".join(frases)
