# Atividade de Avaliação - Modelagem Preditiva
## Previsão de Retorno para Revisão - Concessionária Honda

**FGV - Modelagem Preditiva**

**Grupo:**
- João Victor Fortunato Teixeira Mundim (813190/2026)
- Martinho Aloisio Lenz (812869/2026)
- Rodrigo Pimentel Carneiro (812440/2026)
- Bruno Barradas

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
| Gradient Boosting | 76,8% | **0,7423** | 54,2% | 38,5% | 45,0% | 76,1% ±1,2% |
| Random Forest | **77,1%** | 0,7389 | **55,8%** | 35,2% | 43,2% | **76,4%** ±0,9% |
| Regressão Logística | 75,9% | 0,7156 | 51,3% | **41,2%** | **45,7%** | 75,8% ±1,1% |
| Árvore de Decisão | 74,2% | 0,6834 | 47,1% | 33,8% | 39,4% | 73,9% ±1,5% |

### 5.2 Features Mais Importantes (Random Forest)

| Ranking | Variável | Importância | Interpretação |
|---------|----------|-------------|---------------|
| 1° | VALOR_VENDIDO | 0,1856 | Valor da moto influencia retorno |
| 2° | PRECO_TABELA | 0,1523 | Faixa de preço correlaciona com perfil |
| 3° | PARCELA | 0,1187 | Clientes financiados têm outro perfil |
| 4° | MOTO_MODELO | 0,0954 | Alguns modelos retornam mais |
| 5° | ONDE_MOTO_VENDIDA | 0,0821 | Diferença por loja/região |

### 5.3 Insights de Negócio

- **Clientes COM financiamento** têm taxa de retorno ligeiramente maior
- **Motos da família Utility** têm menor retorno que Leisure
- **Lojas do Gama e Valparaíso** têm melhores taxas de retorno
- **Compradores de kit de acessórios** tendem a retornar mais

---

## 6. Resumo Final e Roteiro para Produção

### 6.1 Modelo Recomendado

**Gradient Boosting** - maior capacidade discriminativa (AUC-ROC = 0,7423)

Alternativa: **Regressão Logística** - se priorizar Recall (identificar mais clientes em risco)

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

- **Identificação geral:** ~77% de acerto
- **Dos identificados como "vai retornar":** ~55% efetivamente retornam
- **Melhoria potencial:** Aumento de 5-10% na taxa de retorno com ações direcionadas

### 6.4 Limitações

- Modelo baseado em dados de um período específico (2025)
- Não considera fatores externos (sazonalidade, campanhas)
- Recomenda-se retreinamento semestral

---

*Documento elaborado com auxílio de ferramentas de IA para geração de código e análise. As decisões de modelagem e interpretações foram conduzidas pelo grupo.*
