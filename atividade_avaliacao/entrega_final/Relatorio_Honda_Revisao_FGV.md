# Atividade de Avaliação - Modelagem Preditiva
## Previsão de Retorno para Revisão - Concessionária Honda

**FGV - Modelagem Preditiva**

**Grupo:**
- João Victor Fortunato Teixeira Mundim (813190/2026)
- Martinho Aloisio Lenz (812869/2026)
- Nicolas Henrique Richardt da Silva (813138/2026)
- Renato Vargas Monteiro (812872/2026)
- Rodrigo Pimentel Carneiro (812440/2026)

---

## 1. Enunciado do Problema

### Contexto de Negócio

O problema abordado é a **previsão de retorno de clientes para revisão programada** em uma concessionária Honda. A revisão é um momento crucial no pós-venda, pois:

- Gera receita de serviços e peças
- Fortalece o relacionamento com o cliente
- Aumenta a chance de futuras vendas (troca de moto)
- Identifica problemas antes de reclamações

### Problema/Oportunidade

**Pergunta de negócio:** *Quais clientes têm maior probabilidade de NÃO retornar para revisão?*

A identificação antecipada de clientes em risco permite ações proativas de contato, ofertas de desconto e agendamento prévio.

### Família de Modelagem

**Classificação binária** - onde:
- **Classe 0:** Cliente NÃO retornou para revisão
- **Classe 1:** Cliente RETORNOU para revisão

### Dataset Utilizado

| Característica | Valor |
|----------------|-------|
| **Fonte** | Dados reais da concessionária (anonimizados) |
| **Registros** | 10.467 vendas de motocicletas |
| **Período** | Janeiro a Dezembro de 2025 |
| **Variáveis** | 16 (15 preditoras + 1 alvo) |
| **Distribuição** | 73,9% Não Retornou / 26,1% Retornou |

**Limitação:** O dataset não contém data de extração/corte nem horizonte de acompanhamento documentado. Os resultados assumem que o rótulo RETORNOU_REVISAO representa adequadamente o comportamento de retorno.

---

## 2. Engenharia de Variáveis (Feature Engineering)

### 2.1 Seleção de Variáveis

**Variáveis utilizadas no modelo:**

| Variável | Tipo | Descrição |
|----------|------|-----------|
| MOTO_MODELO | Categórica | Modelo da motocicleta (113 categorias) |
| MOTO_FAMILIA | Categórica | Família (12 categorias) |
| COR | Categórica | Cor simplificada (7 categorias) |
| ANO_MODELO | Numérica | Ano do modelo |
| PRECO_TABELA | Numérica | Preço de tabela |
| VALOR_VENDIDO | Numérica | Valor efetivamente vendido |
| DESCONTO_INCONDICIONAL | Numérica | Valor do desconto concedido |
| PERC_DESCONTO | Numérica | Percentual de desconto (variável derivada) |
| TEM_FINANCIAMENTO | Binária | Se foi financiada (0/1) |
| FINANCEIRA_RESUMIDA | Categórica | Banco financiador |
| MESES | Numérica | Prazo do financiamento |
| PARCELA | Numérica | Valor da parcela |
| TEM_KIT | Binária | Se comprou kit de acessórios |
| TEM_VOUCHER | Binária | Se usou voucher promocional |
| ONDE_MOTO_VENDIDA | Categórica | Cidade/loja da venda |

**Total: 5 variáveis categóricas e 10 variáveis numéricas.**

**Variáveis removidas:** CHASSI_COMPLETO (identificador), números de notas fiscais e códigos internos.

### 2.2 Tratamento de Dados

| Técnica | Aplicação |
|---------|-----------|
| **Valores ausentes** | Preenchidos com 0 para flags e "SEM_FINANCIAMENTO" para financeira |
| **Simplificação de cores** | Agrupamento de 46 cores em 7 categorias principais |
| **Variável derivada** | PERC_DESCONTO = desconto / preço_tabela × 100 |
| **Label Encoding** | Para modelos de árvore (ajustado apenas no treino) |
| **One-Hot Encoding** | Para Regressão Logística (+ StandardScaler ajustado no treino) |

**Nota:** O pré-processamento (encoding e scaling) foi aplicado separadamente em treino e teste para evitar vazamento de dados.

---

## 3. Modelagens Executadas

Foram implementados os **4 modelos obrigatórios** para classificação:

| # | Modelo | Tipo | Critério |
|---|--------|------|----------|
| 1 | Árvore de Decisão | Base | CART com Entropia |
| 2 | Regressão Logística | Base | L2 (Ridge) |
| 3 | Random Forest | Ensemble (Bagging) | Entropia |
| 4 | Gradient Boosting | Ensemble (Boosting) | Deviance |

**Particionamento:** 80% treino (8.373 registros) / 20% teste (2.094 registros)  
**Validação:** Stratified split + 5-fold Cross-Validation (apenas no treino, com shuffle)

---

## 4. Configuração dos Modelos

### 4.1 Árvore de Decisão (CART/Entropia)

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| criterion | entropy | Ganho de informação |
| max_depth | 5 | Evita overfitting |
| min_samples_split | 20 | Poda mais agressiva |
| min_samples_leaf | 10 | Generalização |

### 4.2 Regressão Logística

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| penalty | l2 | Regularização Ridge |
| C | 1.0 | Regularização padrão |
| solver | lbfgs | Eficiente para L2 |

### 4.3 Random Forest

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| n_estimators | 100 | Trade-off performance/custo |
| max_depth | 10 | Evita overfitting |
| max_features | sqrt | Recomendado para classificação |

### 4.4 Gradient Boosting

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| n_estimators | 100 | Equilíbrio |
| learning_rate | 0.1 | Taxa padrão |
| max_depth | 3 | Árvores rasas para boosting |
| subsample | 0.8 | Stochastic GB |

---

## 5. Resultados

### 5.1 Quadro Comparativo de Desempenho

**Baseline (sempre prever classe 0):** 73,93%

| Modelo | Acurácia | AUC-ROC | Precisão(1) | Recall(1) | Recall(0) | F1(1) | CV Treino |
|--------|----------|---------|-------------|-----------|-----------|-------|-----------|
| **Regressão Logística** | **76,84%** | **0,8072** | 58,18% | **39,74%** | 89,92% | **47,23%** | 75,68% ±0,61% |
| Random Forest | 76,79% | 0,8071 | **61,19%** | 30,04% | **93,28%** | 40,29% | 75,41% ±0,36% |
| Gradient Boosting | 76,27% | 0,8062 | 57,40% | 34,80% | 90,89% | 43,33% | 75,58% ±0,56% |
| Árvore de Decisão | 74,98% | 0,7546 | 53,96% | 27,47% | 91,73% | 36,41% | 74,55% ±0,94% |

**Nota sobre métricas:**
- **Recall(1):** Proporção de clientes que RETORNARAM e foram corretamente identificados
- **Recall(0):** Proporção de clientes que NÃO RETORNARAM e foram corretamente identificados
- Para identificar clientes em risco de NÃO retornar, usar probabilidade = 1 - P(retorno)

### 5.2 Features Mais Importantes (Random Forest)

| Ranking | Variável | Importância |
|---------|----------|-------------|
| 1° | DESCONTO_INCONDICIONAL | 0,1156 |
| 2° | VALOR_VENDIDO | 0,1005 |
| 3° | PRECO_TABELA | 0,0933 |
| 4° | PERC_DESCONTO | 0,0915 |
| 5° | PARCELA | 0,0791 |

### 5.3 Insights de Negócio

- **Clientes COM financiamento** têm taxa de retorno significativamente maior (40,85% vs 15,39% - diferença de 25 p.p.)
- **Clientes com kit de acessórios** retornam mais (45,31% vs 17,22%)
- **Por loja:** Recanto das Emas (39,75%) e Valparaíso (37,58%) têm melhores taxas; Gama (19,26%) tem a menor taxa

**Nota:** Estas são associações observadas nos dados. Não demonstram relação causal.

---

## 6. Resumo Final e Roteiro para Produção

### 6.1 Modelo Recomendado

**Regressão Logística** - melhor AUC-ROC (0,8072) e melhor equilíbrio entre precisão e recall.

Os modelos superam o baseline em ~3 p.p. de acurácia. A diferença entre os três melhores modelos é pequena (< 0,001 em AUC), sugerindo que a escolha pode considerar também interpretabilidade e custo computacional.

### 6.2 Roteiro de Execução em Produção

1. **Entrada:** Dados da venda no momento do faturamento
2. **Pré-processamento:** One-Hot Encoding + StandardScaler (usando transformadores do treino)
3. **Predição:** Modelo retorna probabilidade de retorno (0 a 100%)
4. **Risco de não retorno:** 100% - probabilidade de retorno

| Probabilidade de Retorno | Risco | Ação Recomendada |
|--------------------------|-------|------------------|
| < 25% | **ALTO** | Contato imediato, oferta de desconto na revisão |
| 25% - 40% | **MÉDIO** | Lembrete reforçado (WhatsApp + ligação) |
| > 40% | **BAIXO** | Lembrete padrão por SMS |

### 6.3 Limitações

- Modelo baseado em dados de um período específico (2025)
- Não há documentação da janela de observação do alvo
- Não considera fatores externos (sazonalidade, campanhas)
- Recomenda-se validação com piloto controlado antes de escalar

---

*Documento elaborado com auxílio de ferramentas de IA para geração de código e análise. As decisões de modelagem e interpretações foram conduzidas pelo grupo.*
