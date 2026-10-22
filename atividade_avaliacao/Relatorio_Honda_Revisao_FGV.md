# Atividade de Avaliação - Modelagem Preditiva
## Previsão de Retorno para Revisão - Concessionária Honda

**FGV - Modelagem Preditiva**

**Grupo:**
- João Victor Fortunato Teixeira Mundim (813190/2026)
- Rodrigo Pimentel Carneiro (812440/2026)

---

## 1. Enunciado do Problema

### Contexto de Negócio

O problema abordado é a **previsão de retorno de clientes para a primeira revisão programada** em uma concessionária Honda. A revisão é um momento crucial no pós-venda, pois:

- Gera receita de serviços e peças
- Fortalece o relacionamento com o cliente
- Aumenta a chance de futuras vendas (troca de moto)
- Identifica problemas antes de reclamações

### Problema/Oportunidade

**Pergunta de negócio:** *Quais clientes têm maior probabilidade de NÃO retornar para a primeira revisão?*

A identificação antecipada de clientes em risco permite:
- Ações proativas de contato (ligação, WhatsApp)
- Ofertas de desconto ou facilidades
- Agendamento prévio no momento da venda
- Aumento da taxa de retenção no pós-venda

### Família de Modelagem

**Classificação binária** - onde:
- **Classe 0:** Cliente NÃO retornou para revisão
- **Classe 1:** Cliente RETORNOU para revisão

### Dataset Utilizado

| Característica | Valor |
|----------------|-------|
| **Fonte** | Dados reais da concessionária (anonimizados) |
| **Registros** | 10.467 vendas de motocicletas |
| **Período** | Janeiro a Outubro de 2025 |
| **Variáveis** | 16 (15 preditoras + 1 alvo) |
| **Distribuição** | 73,9% Não Retornou / 26,1% Retornou |

---

## 2. Engenharia de Variáveis (Feature Engineering)

### 2.1 Seleção de Variáveis

**Variáveis utilizadas no modelo:**

| Variável | Tipo | Descrição |
|----------|------|-----------|
| MOTO_MODELO | Categórica | Modelo da motocicleta (CG 160, Bros, Biz, etc.) |
| MOTO_FAMILIA | Categórica | Família (Utility, Entry, Leisure, Scooter, etc.) |
| COR | Categórica | Cor simplificada (6 categorias) |
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

**Variáveis removidas (dados sensíveis ou sem valor preditivo):**
- CHASSI_COMPLETO (identificador)
- Números de notas fiscais e códigos internos
- Dados de revisões futuras (evitar data leakage)

### 2.2 Tratamento de Dados

| Técnica | Aplicação |
|---------|-----------|
| **Valores ausentes** | Preenchidos com 0 para flags (TEM_*) e "SEM_FINANCIAMENTO" para financeira |
| **Simplificação de cores** | Agrupamento de 46 cores em 7 categorias principais |
| **Variável derivada** | PERC_DESCONTO = desconto / preço_tabela × 100 |
| **Label Encoding** | Para modelos de árvore (6 variáveis categóricas) |
| **One-Hot Encoding** | Para Regressão Logística (+ StandardScaler) |

---

## 3. Modelagens Executadas

Foram implementados os **4 modelos obrigatórios** para classificação:

| # | Modelo | Tipo | Critério |
|---|--------|------|----------|
| 1 | Árvore de Decisão | Base | Entropia (ID3) |
| 2 | Regressão Logística | Base | L2 (Ridge) |
| 3 | Random Forest | Ensemble (Bagging) | Entropia |
| 4 | Gradient Boosting | Ensemble (Boosting) | Deviance |

**Particionamento:** 80% treino (8.373 registros) / 20% teste (2.094 registros)  
**Validação:** Stratified split + 5-fold Cross-Validation

---

## 4. Tentativas de Otimização

### 4.1 Árvore de Decisão

| Parâmetro | Valores Testados | Escolhido | Justificativa |
|-----------|------------------|-----------|---------------|
| criterion | gini, entropy | **entropy** | Melhor AUC-ROC |
| max_depth | 3, 5, 7, 10 | **5** | Evita overfitting |
| min_samples_split | 5, 10, 20 | **20** | Poda mais agressiva |
| min_samples_leaf | 5, 10, 20 | **10** | Generalização |

### 4.2 Regressão Logística

| Parâmetro | Valores Testados | Escolhido | Justificativa |
|-----------|------------------|-----------|---------------|
| penalty | l1, l2 | **l2** | Mais estável |
| C | 0.1, 1.0, 10 | **1.0** | Regularização padrão |

### 4.3 Random Forest

| Parâmetro | Valores Testados | Escolhido | Justificativa |
|-----------|------------------|-----------|---------------|
| n_estimators | 50, 100, 200 | **100** | Trade-off performance/custo |
| max_depth | 5, 10, None | **10** | Evita overfitting |
| max_features | sqrt, log2 | **sqrt** | Recomendado para classificação |

### 4.4 Gradient Boosting

| Parâmetro | Valores Testados | Escolhido | Justificativa |
|-----------|------------------|-----------|---------------|
| n_estimators | 50, 100, 150 | **100** | Equilíbrio |
| learning_rate | 0.05, 0.1, 0.2 | **0.1** | Taxa padrão |
| max_depth | 3, 5, 7 | **3** | Árvores rasas para boosting |
| subsample | 0.8, 1.0 | **0.8** | Stochastic GB |

---

## 5. Resultados

### 5.1 Quadro Comparativo de Desempenho

| Modelo | Acurácia | AUC-ROC | Precisão | Recall | F1-Score | CV (5-fold) |
|--------|----------|---------|----------|--------|----------|-------------|
| Random Forest | **76,84%** | **0,8075** | **61,42%** | 30,04% | 40,34% | 58,35% ±14,74% |
| Regressão Logística | 76,74% | 0,8071 | 57,87% | **39,74%** | **47,12%** | 57,39% ±14,22% |
| Gradient Boosting | 76,31% | 0,8066 | 57,58% | 34,80% | 43,38% | 59,43% ±14,08% |
| Árvore de Decisão | 74,98% | 0,7546 | 53,96% | 27,47% | 36,41% | 60,41% ±12,16% |

### 5.2 Features Mais Importantes (Random Forest)

| Ranking | Variável | Importância | Interpretação |
|---------|----------|-------------|---------------|
| 1° | DESCONTO_INCONDICIONAL | 0,1156 | Desconto oferecido é forte preditor |
| 2° | VALOR_VENDIDO | 0,1005 | Valor da moto influencia retorno |
| 3° | PRECO_TABELA | 0,0933 | Faixa de preço correlaciona com perfil |
| 4° | PERC_DESCONTO | 0,0915 | Percentual de desconto impacta decisão |
| 5° | PARCELA | 0,0791 | Clientes financiados têm outro perfil |
| 6° | MOTO_MODELO | 0,0744 | Alguns modelos retornam mais |
| 7° | ONDE_MOTO_VENDIDA | 0,0733 | Diferença por loja/região |
| 8° | TEM_KIT | 0,0717 | Clientes com acessórios mais engajados |

### 5.3 Insights de Negócio

- **Clientes COM financiamento** têm taxa de retorno ligeiramente maior
- **Motos da família Utility** têm menor retorno que Leisure
- **Lojas do Gama e Valparaíso** têm melhores taxas de retorno
- **Compradores de kit de acessórios** tendem a retornar mais

---

## 6. Resumo Final e Roteiro para Produção

### 6.1 Modelo Recomendado

**Random Forest** - maior capacidade discriminativa (AUC-ROC = 0,8075) e melhor precisão (61,42%)

Alternativa: **Regressão Logística** - se priorizar Recall (39,74%) para identificar mais clientes em risco

### 6.2 Roteiro de Execução em Produção

1. **Entrada:** Dados da venda no momento do faturamento
2. **Pré-processamento:** Label Encoding nas categóricas
3. **Predição:** Modelo retorna probabilidade de retorno (0 a 100%)
4. **Ação baseada no resultado:**

| Probabilidade de Retorno | Risco | Ação Recomendada |
|--------------------------|-------|------------------|
| < 20% | **ALTO** | Contato imediato, oferta de desconto na revisão |
| 20% - 40% | **MÉDIO** | Lembrete reforçado (WhatsApp + ligação) |
| > 40% | **BAIXO** | Lembrete padrão por SMS |

### 6.3 Desempenho Esperado

- **Identificação geral:** ~77% de acerto (acurácia do Random Forest)
- **Dos identificados como "vai retornar":** ~61% efetivamente retornam (precisão)
- **Capacidade discriminativa:** AUC-ROC de 0,81 indica boa separação entre classes
- **Melhoria potencial:** Aumento de 5-10% na taxa de retorno com ações direcionadas

### 6.4 Limitações

- Modelo baseado em dados de um período específico (2025)
- Não considera fatores externos (sazonalidade, campanhas)
- Recomenda-se retreinamento semestral

---

*Documento elaborado com auxílio de ferramentas de IA para geração de código e análise. As decisões de modelagem e interpretações foram conduzidas pelo grupo.*
