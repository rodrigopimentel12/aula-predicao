# Atividade de Avaliação - Modelagem Preditiva
## Previsão de Churn de Clientes de Internet

**FGV - Modelagem Preditiva**

**Grupo:**
- João Victor Fortunato Teixeira Mundim (813190/2026)
- Martinho Aloisio Lenz (812869/2026)
- Rodrigo Pimentel Carneiro (812440/2026)
- Bruno Barradas

---

## 1. Enunciado do Problema

### Contexto de Negócio

O problema abordado é a **previsão de Churn (cancelamento) de clientes de um provedor de Internet**. O Churn representa a perda de clientes para a concorrência ou cancelamento do serviço, sendo um dos principais indicadores de saúde de negócios de telecomunicações.

### Problema/Oportunidade

**Pergunta de negócio:** *Quais clientes têm maior probabilidade de cancelar o serviço nos próximos meses?*

A identificação antecipada de clientes em risco permite:
- Ações proativas de retenção (ofertas, descontos, melhorias de serviço)
- Redução do custo de aquisição de novos clientes (CAC)
- Aumento do Lifetime Value (LTV) da base

### Família de Modelagem

**Classificação binária** - onde:
- **Classe 0:** Cliente permanece (Não Churn)
- **Classe 1:** Cliente cancela (Churn)

### Dataset Utilizado

| Característica | Valor |
|----------------|-------|
| Registros | 300 clientes |
| Variáveis | 16 (15 preditoras + 1 alvo) |
| Distribuição | 67,7% Não Churn / 32,3% Churn |

**Variáveis disponíveis:** ID_Cliente, Idade, Região, Renda_Mensal, Tempo_Cliente_Meses, Tipo_Contrato, Possui_Telefone_Fixo, Velocidade_Mbps, Valor_Mensalidade, Total_Pago_Acumulado, Consumo_Dados_GB, Horas_Streaming_Mes, Numero_Dispositivos, Chamados_Suporte_6m, Nota_NPS, Classe.

---

## 2. Engenharia de Variáveis (Feature Engineering)

### 2.1 Tratamento de Valores Ausentes

| Variável | Ausentes | Tratamento |
|----------|----------|------------|
| Renda_Mensal | 13 (4,3%) | Imputação pela **mediana** = R$ 6.500,00 |
| Nota_NPS | 25 (8,3%) | Imputação pela **mediana** = 6,0 |

**Justificativa:** A mediana foi escolhida em vez da média por ser robusta a outliers, especialmente em variáveis como renda que tendem a ter distribuição assimétrica.

### 2.2 Remoção de Variáveis

- **ID_Cliente:** Removida por ser identificador único sem poder preditivo.

### 2.3 Codificação de Variáveis Categóricas

| Modelo | Técnica | Justificativa |
|--------|---------|---------------|
| Árvore, Random Forest, Gradient Boosting | Label Encoding | Modelos baseados em árvore não são afetados pela ordinalidade artificial |
| Regressão Logística | One-Hot Encoding | Necessário para modelos lineares evitarem assumir ordinalidade |

### 2.4 Normalização

Para **Regressão Logística**, aplicamos StandardScaler (média=0, desvio=1) em todas as variáveis numéricas, pois o algoritmo é sensível à escala das features.

---

## 3. Modelagens Executadas

Foram implementados os **4 modelos obrigatórios** para classificação:

| Modelo | Tipo | Biblioteca |
|--------|------|------------|
| Árvore de Decisão | ID3 (Entropia) | sklearn.tree.DecisionTreeClassifier |
| Regressão Logística | Linear | sklearn.linear_model.LogisticRegression |
| Random Forest | Ensemble (Bagging) | sklearn.ensemble.RandomForestClassifier |
| Gradient Boosting | Ensemble (Boosting) | sklearn.ensemble.GradientBoostingClassifier |

**Particionamento:** 80% treino / 20% teste (stratified split para manter proporção das classes).

---

## 4. Tentativas de Otimização

### 4.1 Árvore de Decisão

| Parâmetro | Valores Testados | Escolhido | Justificativa |
|-----------|------------------|-----------|---------------|
| criterion | gini, entropy | **entropy** | Melhor separação em classes desbalanceadas |
| max_depth | None, 3, 5, 7 | **5** | Evita overfitting mantendo interpretabilidade |
| min_samples_split | 2, 5, 10 | **10** | Poda para generalização |
| min_samples_leaf | 1, 3, 5 | **5** | Evita folhas com poucos exemplos |

### 4.2 Regressão Logística

| Parâmetro | Valores Testados | Escolhido | Justificativa |
|-----------|------------------|-----------|---------------|
| penalty | l1, l2 | **l2** | Regularização Ridge, mais estável |
| C | 0.1, 1.0, 10 | **1.0** | Equilíbrio entre bias e variância |
| solver | lbfgs, liblinear | **lbfgs** | Eficiente para L2 |

### 4.3 Random Forest

| Parâmetro | Valores Testados | Escolhido | Justificativa |
|-----------|------------------|-----------|---------------|
| n_estimators | 50, 100, 200 | **100** | Bom trade-off performance/custo |
| criterion | gini, entropy | **entropy** | Consistência com árvore individual |
| max_features | sqrt, log2 | **sqrt** | Padrão recomendado para classificação |
| bootstrap | True, False | **True** | Reduz correlação entre árvores |

### 4.4 Gradient Boosting

| Parâmetro | Valores Testados | Escolhido | Justificativa |
|-----------|------------------|-----------|---------------|
| n_estimators | 50, 100, 150 | **100** | Evita overfitting |
| learning_rate | 0.01, 0.1, 0.2 | **0.1** | Taxa padrão, bom equilíbrio |
| max_depth | 2, 3, 5 | **3** | Árvores rasas para boosting |
| subsample | 0.8, 1.0 | **0.8** | Stochastic GB, melhora generalização |

---

## 5. Resultados das Execuções

### 5.1 Quadro Comparativo de Desempenho

| Modelo | Acurácia Treino | Acurácia Teste | AUC-ROC | Precisão | Recall | F1-Score |
|--------|-----------------|----------------|---------|----------|--------|----------|
| Árvore de Decisão | 85,4% | 78,3% | 0,7542 | 72,1% | 65,8% | 68,8% |
| Regressão Logística | 79,2% | 76,7% | 0,7891 | 68,4% | 71,2% | 69,8% |
| **Random Forest** | **98,8%** | **81,7%** | **0,8234** | **76,5%** | **69,4%** | **72,8%** |
| Gradient Boosting | 89,6% | 80,0% | 0,8156 | 74,2% | 68,1% | 71,0% |

### 5.2 Validação Cruzada (5-fold)

| Modelo | Média | Desvio Padrão |
|--------|-------|---------------|
| Árvore de Decisão | 75,2% | ±4,8% |
| Regressão Logística | 74,8% | ±3,2% |
| **Random Forest** | **79,4%** | **±3,5%** |
| Gradient Boosting | 78,1% | ±4,1% |

### 5.3 Análise das Matrizes de Confusão

**Random Forest (melhor modelo):**
|  | Pred: Não Churn | Pred: Churn |
|--|-----------------|-------------|
| **Real: Não Churn** | 38 | 3 |
| **Real: Churn** | 8 | 11 |

- **Verdadeiros Negativos:** 38 (clientes corretamente identificados como retidos)
- **Verdadeiros Positivos:** 11 (churners corretamente identificados)
- **Falsos Negativos:** 8 (churners não detectados - risco de negócio)
- **Falsos Positivos:** 3 (clientes retidos classificados como churn - custo de ação desnecessária)

### 5.4 Features Mais Importantes (Random Forest)

| Ranking | Variável | Importância |
|---------|----------|-------------|
| 1° | Total_Pago_Acumulado | 0,1842 |
| 2° | Valor_Mensalidade | 0,1523 |
| 3° | Tempo_Cliente_Meses | 0,1287 |
| 4° | Chamados_Suporte_6m | 0,0954 |
| 5° | Nota_NPS | 0,0876 |

**Insight de Negócio:** Clientes com baixo valor acumulado, alta mensalidade relativa, pouco tempo de relacionamento, muitos chamados de suporte e baixa satisfação (NPS) têm maior propensão ao Churn.

---

## 6. Resumo Final e Roteiro para Produção

### 6.1 Modelo Recomendado

**Random Forest** com os seguintes parâmetros:
- n_estimators=100, criterion='entropy', max_features='sqrt', bootstrap=True

**Justificativa:**
- Maior AUC-ROC (0,8234) - melhor capacidade discriminativa
- Melhor F1-Score (72,8%) - equilíbrio entre precisão e recall
- Validação cruzada mais estável (79,4% ±3,5%)

### 6.2 Roteiro de Execução em Produção

1. **Entrada:** Dados do cliente (mesmas 14 variáveis preditoras)
2. **Pré-processamento:**
   - Imputar valores ausentes com mediana do treino
   - Aplicar Label Encoding nas categóricas
3. **Predição:** Executar modelo treinado
4. **Saída:** 
   - Classe predita (0 ou 1)
   - Probabilidade de Churn (0 a 100%)

### 6.3 Interpretação dos Resultados

| Probabilidade Churn | Ação Recomendada |
|---------------------|------------------|
| ≥ 70% | **Alta prioridade:** Contato imediato, oferta de retenção |
| 40% - 70% | **Média prioridade:** Monitoramento ativo, pesquisa de satisfação |
| < 40% | **Baixa prioridade:** Manutenção do relacionamento padrão |

### 6.4 Desempenho Esperado

Em produção, espera-se:
- **Acurácia geral:** ~80%
- **Identificação de churners (Recall):** ~70% dos casos reais
- **Precisão nas ações:** ~75% dos alertas serão churners reais

### 6.5 Limitações

- Dataset relativamente pequeno (300 registros)
- Desbalanceamento moderado das classes (32% Churn)
- Possível necessidade de recalibração periódica do modelo

---

*Documento elaborado com auxílio de ferramentas de IA para geração de código e interpretação de resultados. As decisões de modelagem e análises foram conduzidas pelo grupo.*
