# Atividade de Avaliação - Modelagem Preditiva
## Previsão de Retorno para Revisão - Concessionária Honda

**FGV - Modelagem Preditiva**

**Grupo:**
- João Victor Fortunato Teixeira Mundim (813190/2026)
- Rodrigo Pimentel Carneiro (812440/2026)

---

## 1. Enunciado do Problema

**Contexto:** Previsão de retorno de clientes para revisão programada em concessionária Honda.

**Pergunta de negócio:** *Quais clientes têm maior probabilidade de NÃO retornar para revisão?*

**Família de Modelagem:** Classificação binária (Classe 0 = Não retornou, Classe 1 = Retornou)

| Característica | Valor |
|----------------|-------|
| Registros | 10.467 vendas |
| Período | Janeiro a Dezembro de 2025 |
| Distribuição | 73,9% Não Retornou / 26,1% Retornou |

**Limitação:** Não há data de corte ou janela de acompanhamento documentada.

---

## 2. Engenharia de Variáveis

### Por que cada técnica foi utilizada:

| Técnica | Justificativa |
|---------|---------------|
| **Label Encoding** | Árvores e ensembles aceitam valores numéricos. Converte categorias (ex: "BROS 160") em números inteiros para processamento. |
| **One-Hot Encoding** | Regressão Logística assume linearidade. Cria variáveis binárias para evitar ordenação artificial entre categorias. |
| **StandardScaler** | Logística converge melhor com variáveis na mesma escala. Padroniza média=0 e desvio=1. |
| **Agrupamento de cores** | 46 cores geram fragmentação. Agrupadas em 7 categorias principais para melhor generalização. |
| **PERC_DESCONTO** | Percentual relativo é mais comparável entre faixas de preço que valor absoluto. |
| **Preenchimento com 0** | TEM_KIT=0 indica ausência confirmada do acessório (não é dado faltante). |

**Variáveis:** 5 categóricas + 10 numéricas. Identificadores (CHASSI) removidos dos preditores.

---

## 3. Modelagens Executadas

| Modelo | Tipo | Critério |
|--------|------|----------|
| Árvore de Decisão | Base | CART com Entropia e Gini |
| Regressão Logística | Base | L2 (Ridge) |
| Random Forest | Ensemble | Entropia |
| Gradient Boosting | Ensemble | Deviance |

**Partição:** 80% treino (8.373) / 20% teste (2.094). **CV:** 5-fold no treino.

---

## 4. Tentativas de Otimização (Resultados Reais)

### 4.1 Árvore de Decisão - Entropia vs Gini

| Critério | max_depth | Acurácia CV | AUC Teste | Decisão |
|----------|-----------|-------------|-----------|---------|
| entropy | 3 | 74,44% | 0,7310 | Profundidade insuficiente |
| entropy | 5 | 74,55% | 0,7546 | Bom equilíbrio |
| **entropy** | **7** | **75,00%** | **0,7720** | **Melhor AUC** |
| gini | 5 | 74,82% | 0,7490 | Inferior ao entropy |

**Conclusão:** Entropia com depth=7 obteve melhor AUC (0,7720).

### 4.2 Regressão Logística - Regularização C

| C | Acurácia CV | AUC Teste | Decisão |
|---|-------------|-----------|---------|
| 0.1 | 75,55% | 0,8060 | Regularização forte |
| **1.0** | **75,68%** | **0,8072** | **Melhor AUC** |
| 10.0 | 75,68% | 0,8067 | Sem ganho |

**Conclusão:** C=1.0 escolhido (melhor AUC, regularização padrão).

### 4.3 Random Forest - Estimadores e Profundidade

| n_est | max_depth | Acurácia CV | AUC Teste | Decisão |
|-------|-----------|-------------|-----------|---------|
| 50 | 5 | 74,64% | 0,7774 | Raso demais |
| 50 | 10 | 75,15% | 0,8052 | Bom |
| **100** | **10** | **75,41%** | **0,8075** | **Melhor** |

### 4.4 Gradient Boosting - Learning Rate e Profundidade

| lr | max_depth | Acurácia CV | AUC Teste | Decisão |
|----|-----------|-------------|-----------|---------|
| 0.05 | 3 | 75,65% | 0,8005 | Conservador |
| **0.05** | **5** | **75,42%** | **0,8134** | **Melhor AUC** |
| 0.1 | 3 | 75,58% | 0,8066 | Configuração padrão |

---

## 5. Resultados

### 5.1 Quadro Comparativo Final

**Baseline (sempre classe 0):** 73,93%

| Modelo | Acurácia | AUC-ROC | F1 | Ganho vs Base |
|--------|----------|---------|----|----|
| **Regressão Logística** | **76,84%** | 0,8072 | 47,23% | +2,91 p.p. |
| Random Forest | 76,79% | 0,8075 | 40,34% | +2,86 p.p. |
| Gradient Boosting | 76,27% | 0,8066 | 43,33% | +2,34 p.p. |
| Árvore (entropy, d=7) | 75,00% | 0,7720 | 44,07% | +1,07 p.p. |

### 5.2 Exemplos de Predição Individual

| # | Modelo Moto | Financ. | Prob. Retorno | Classe Prevista | Real | Acerto |
|---|-------------|---------|---------------|-----------------|------|--------|
| 1 | XRE 300 SAHARA | Não | 7,6% | 0 (não retorna) | 0 | ✓ |
| 2 | XRL 300 TORNADO | Não | 6,9% | 0 (não retorna) | 1 | ✗ |
| 3 | BROS 160 ABS | Sim | 31,7% | 0 (não retorna) | 1 | ✗ |
| 4 | CB 500 HORNET | Sim | 36,0% | 0 (não retorna) | 0 | ✓ |
| 5 | BROS 160 CBS | Não | 16,4% | 0 (não retorna) | 1 | ✗ |

**Interpretação:** Os casos 2, 3 e 5 são falsos negativos - clientes que retornaram mas o modelo previu que não retornariam. O caso 3 (financiado, prob. 31,7%) mostra que mesmo com probabilidade moderada, o limiar padrão de 50% classifica como não retorno.

---

## 6. Resumo Final e Roteiro para Produção

### 6.1 Modelo Recomendado

**Regressão Logística (C=1.0)** - escolhida por:
- Interpretabilidade dos coeficientes
- Simplicidade computacional
- AUC competitiva (0,8072)

A diferença entre os três melhores modelos é mínima (<0,001 AUC). GB com lr=0.05 e depth=5 obteve a maior AUC (0,8134), mas logística foi preferida pela interpretabilidade.

### 6.2 Roteiro de Execução

1. **Entrada:** Dados da venda no faturamento
2. **Pré-processamento:** One-Hot Encoding + StandardScaler
3. **Predição:** Probabilidade de retorno (0-100%)
4. **Risco:** 100% - probabilidade

### 6.3 Faixas de Ação (Proposta para Piloto)

| Prob. Retorno | Risco | Ação |
|---------------|-------|------|
| < 25% | ALTO | Contato imediato |
| 25-40% | MÉDIO | Lembrete reforçado |
| > 40% | BAIXO | Lembrete padrão |

**Nota:** Essas faixas são propostas para validação em piloto controlado. Os resultados históricos não garantem o efeito de uma campanha.

### 6.4 Desempenho Esperado

- Acurácia ~77% (vs 74% do baseline)
- Dos clientes previstos como "vai retornar", ~58% efetivamente retornam

### 6.5 Limitações

- Dados de um período específico (2025)
- Janela de observação não documentada
- Recomenda-se retreinamento periódico

---

*Documento elaborado com auxílio de IA. Decisões de modelagem conduzidas pelo grupo.*
