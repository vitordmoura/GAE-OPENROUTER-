"""Proteção de segredos e dados pessoais.

Duas funções principais:
- mascarar(): remove chaves de API de qualquer texto antes de ele ser exibido ou salvo.
- varrer():   procura chaves e dados pessoais nos arquivos do projeto antes da entrega.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

NOME_VARIAVEL_CHAVE = "OPENROUTER_API_KEY"
MASCARA = "[REMOVIDO]"

# Padrões que indicam uma chave real (críticos: bloqueiam a entrega).
PADROES_CRITICOS = {
    "chave OpenRouter": re.compile(r"sk-or-(?:v1-)?[A-Za-z0-9_-]{20,}"),
    "token Bearer": re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{20,}"),
    "atribuição de chave": re.compile(
        r"(?i)openrouter_api_key\s*[=:]\s*['\"]?[A-Za-z0-9_-]{16,}"
    ),
}

# Padrões que merecem revisão manual (avisos: não bloqueiam a entrega).
PADROES_AVISO = {
    "e-mail": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "CPF": re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),
    "telefone": re.compile(r"\(?\b\d{2}\)?\s?9\d{4}-?\d{4}\b"),
}

PASTAS_IGNORADAS = {".git", ".venv", "venv", "env", "__pycache__", ".ipynb_checkpoints", "node_modules"}
EXTENSOES_TEXTO = {".py", ".ipynb", ".md", ".txt", ".json", ".jsonl", ".csv", ".toml",
                   ".cfg", ".ini", ".yaml", ".yml", ".env", ".example", ".sh", ""}


def _segredos_do_ambiente() -> list[str]:
    valor = os.environ.get(NOME_VARIAVEL_CHAVE, "").strip()
    return [valor] if len(valor) >= 8 else []


def mascarar(texto: str | None, segredos: list[str] | tuple[str, ...] = ()) -> str | None:
    """Substitui a chave do ambiente, segredos informados e padrões de chave por [REMOVIDO]."""
    if not texto:
        return texto
    for segredo in [*segredos, *_segredos_do_ambiente()]:
        if segredo:
            texto = texto.replace(segredo, MASCARA)
    for padrao in PADROES_CRITICOS.values():
        texto = padrao.sub(MASCARA, texto)
    return texto


def _trecho_mascarado(linha: str, inicio: int, fim: int) -> str:
    achado = linha[inicio:fim]
    visivel = achado[:6] if len(achado) > 10 else achado[:2]
    return f"{visivel}...({len(achado)} caracteres)"


def _arquivos_do_projeto(raiz: Path):
    for caminho in sorted(raiz.rglob("*")):
        if not caminho.is_file():
            continue
        if any(parte in PASTAS_IGNORADAS for parte in caminho.relative_to(raiz).parts):
            continue
        if caminho.suffix.lower() in EXTENSOES_TEXTO or caminho.name.startswith(".env"):
            yield caminho


def varrer(raiz: str | Path = ".", segredos_extras: tuple[str, ...] = ()) -> dict:
    """Varre o projeto e devolve {'criticos': [...], 'avisos': [...], 'arquivos_verificados': n}.

    - Críticos: chave de API (padrão sk-or-..., Bearer, atribuição literal) ou o valor
      exato da OPENROUTER_API_KEY carregada no ambiente.
    - Avisos: e-mails, CPF, telefone e arquivo .env presente na pasta (revisar à mão).
    Os achados nunca exibem o segredo inteiro.
    """
    raiz = Path(raiz).resolve()
    segredos = [s for s in (*segredos_extras, *_segredos_do_ambiente()) if s]
    criticos, avisos, total = [], [], 0

    for arquivo in _arquivos_do_projeto(raiz):
        total += 1
        relativo = str(arquivo.relative_to(raiz))
        if arquivo.name == ".env":
            avisos.append({"arquivo": relativo, "linha": None, "tipo": "arquivo .env",
                           "trecho": "não inclua este arquivo na entrega"})
            continue  # o .env existe justamente para guardar a chave; não é varrido linha a linha
        try:
            linhas = arquivo.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for numero, linha in enumerate(linhas, start=1):
            for segredo in segredos:
                if segredo in linha:
                    i = linha.index(segredo)
                    criticos.append({"arquivo": relativo, "linha": numero,
                                     "tipo": "valor exato da chave do ambiente",
                                     "trecho": _trecho_mascarado(linha, i, i + len(segredo))})
            for tipo, padrao in PADROES_CRITICOS.items():
                for m in padrao.finditer(linha):
                    criticos.append({"arquivo": relativo, "linha": numero, "tipo": tipo,
                                     "trecho": _trecho_mascarado(linha, m.start(), m.end())})
            for tipo, padrao in PADROES_AVISO.items():
                for m in padrao.finditer(linha):
                    avisos.append({"arquivo": relativo, "linha": numero, "tipo": tipo,
                                   "trecho": _trecho_mascarado(linha, m.start(), m.end())})

    return {"criticos": criticos, "avisos": avisos, "arquivos_verificados": total}


def imprimir_relatorio(resultado: dict) -> bool:
    """Mostra o relatório da varredura. Devolve True quando não há achados críticos."""
    print(f"Arquivos verificados: {resultado['arquivos_verificados']}")
    if resultado["criticos"]:
        print(f"\n[ERRO] {len(resultado['criticos'])} possível(is) segredo(s) encontrado(s):")
        for a in resultado["criticos"]:
            print(f"   - {a['arquivo']}:{a['linha']}  [{a['tipo']}]  {a['trecho']}")
    else:
        print("[OK] Nenhuma chave de API encontrada.")
    if resultado["avisos"]:
        print(f"\n[AVISO] {len(resultado['avisos'])} item(ns) para revisão manual:")
        for a in resultado["avisos"]:
            local = f"{a['arquivo']}:{a['linha']}" if a["linha"] else a["arquivo"]
            print(f"   - {local}  [{a['tipo']}]  {a['trecho']}")
    return not resultado["criticos"]
