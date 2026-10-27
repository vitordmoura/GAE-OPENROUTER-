# Comparação entre dois modelos gratuitos do OpenRouter

**Aluno:** Vitor de Moura Nascimento (RM 553806)  
**Turma:** 4ESPS  
**Modelos comparados:** `nvidia/nemotron-3-ultra-550b-a55b:free` (A) e `dots-studio/dots-3-note-preview:free` (B)  


Neste trabalho, o mesmo prompt foi enviado, com os mesmos parâmetros, para dois modelos marcados como gratuitos no catálogo do OpenRouter no momento da execução. As respostas foram comparadas com métricas objetivas (formato, idioma, limite de palavras, acerto de um cálculo com resposta conhecida e quantidade de itens pedidos) e com uma análise escrita das respostas.

## Resultado

Os dois modelos acertaram o conteúdo e cumpriram as 6 regras objetivas nas 3 rodadas. O Nemotron 3 Ultra foi considerado melhor para esta tarefa: respondeu em 22,0 s em média, contra 67,6 s do Dots 3 Note, usou quase seis vezes menos tokens e citou limitações mais ligadas ao uso gratuito por API. O Dots foi um pouco mais didático e detalhou melhor o cálculo, mas com o limite de 4000 tokens não entregou nenhuma resposta. Antes da execução final houve duas tentativas com falha (Gemma 4 31B com provedor sobrecarregado, Inkling Small restrito a aplicativos de agentes e o Dots sem resposta), registradas em `resultados/historico` e discutidas nas seções 9 e 10 do notebook.

## Estrutura

| Arquivo ou pasta | Conteúdo |
|---|---|
| `atividade.ipynb` | Notebook executado, com configuração, catálogo, escolha dos modelos, execução, evidências, métricas, análise, decisões, limitações e conclusão |
| `comparador/openrouter.py` | Consulta do catálogo, chamadas à API com novas tentativas e registro das evidências |
| `comparador/analise.py` | Verificações automáticas das respostas e tabelas comparativas |
| `comparador/seguranca.py` | Remoção de chaves das mensagens e busca de segredos e dados pessoais nos arquivos |
| `verificar_entrega.py` | Verificação final antes do envio; com `--zip`, gera o `entrega.zip` sem o `.env` |
| `resultados/` | Evidências geradas durante a execução |

## Como executar

1. Instalar as dependências (Python 3.10 ou superior): `pip install -r requirements.txt`
2. Configurar a chave no ambiente, nunca no código:
   - Git Bash, Linux ou macOS: `export OPENROUTER_API_KEY="<chave>"`
   - PowerShell: `$env:OPENROUTER_API_KEY="<chave>"`
   - ou criar um arquivo `.env` na pasta do projeto a partir do `.env.example` (o nome tem que ser exatamente `.env`, sem `.txt`)
   - no Google Colab, cadastrar a chave em Secrets com o nome `OPENROUTER_API_KEY`
3. Abrir o `atividade.ipynb` e executar todas as células em ordem.
4. Rodar `python verificar_entrega.py --zip`. O zip só é gerado se nenhuma chave for encontrada no projeto.

## Evidências em `resultados/`

- `catalogo_gratuitos.json`: lista de modelos gratuitos com data e hora da consulta
- `limites_conta.json`: cota diária da conta, sem dados da chave
- `prompt.txt`: prompt usado
- `execucoes.jsonl`: uma linha por chamada, com horário, provedor, id da geração, tokens, tempo de resposta, tentativas e resposta
- `respostas.md`: respostas completas
- `execucoes_tabela.csv` e `comparativo.csv`: tabelas da análise
- `comparativo.png`: gráfico comparativo
- `decisoes_automaticas.json`: trocas de modelo feitas durante a execução
- `ambiente.json`: versões do Python e das bibliotecas
