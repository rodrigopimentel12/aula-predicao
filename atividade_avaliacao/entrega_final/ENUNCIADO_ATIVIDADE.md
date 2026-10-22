# Enunciado da Atividade de Avaliação
## Modelagem Preditiva - FGV

**Data de Entrega:** 22/10/2026

---

## Objetivo

Desenvolver um projeto completo de modelagem preditiva aplicado a um problema de negócio real, utilizando técnicas de classificação aprendidas na disciplina.

---

## Requisitos da Entrega

### 1. Relatório (máximo 6 páginas)

O relatório deve conter:

1. **Enunciado do problema**
   - Contexto de negócio
   - Família de modelagem (classificação/regressão)
   - Descrição do dataset utilizado

2. **Engenharia de Variáveis (Feature Engineering)**
   - Seleção de variáveis
   - Tratamento de valores ausentes
   - Transformações aplicadas (encoding, scaling, etc.)

3. **Modelagens executadas**
   - Implementar **4 modelos obrigatórios**:
     - Árvore de Decisão (com critério de Entropia)
     - Regressão Logística
     - Random Forest
     - Gradient Boosting

4. **Tentativas de otimização**
   - Documentar parâmetros testados
   - Justificar escolhas realizadas

5. **Quadro comparativo de desempenho**
   - Métricas: Acurácia, AUC-ROC, Precisão, Recall, F1-Score
   - Validação Cruzada

6. **Resumo final**
   - Modelo recomendado
   - Roteiro de execução em produção

---

### 2. Código

- Notebook Jupyter (.ipynb) ou script Python (.py)
- Código executável e reproduzível
- Comentários explicativos

---

### 3. Dados

- Dataset utilizado no projeto
- Dados devem estar anonimizados (sem informações pessoais identificáveis)

---

## Formato de Entrega

- Repositório GitHub organizado
- Arquivos em formato Word (.docx) ou PDF

---

## Critérios de Avaliação

- Corretude metodológica (pipeline sem vazamento de dados)
- Qualidade da análise e interpretação
- Clareza do relatório
- Organização do código
- Fundamentação das decisões

---

## Observações Importantes

1. O pré-processamento (encoding, scaling) deve ser aplicado **separadamente** em treino e teste para evitar vazamento de dados
2. A validação cruzada deve ser realizada **apenas no conjunto de treino**
3. As métricas devem ser interpretadas corretamente em relação às classes do problema
4. Limitações do modelo devem ser documentadas

---

*Disciplina: Modelagem Preditiva - FGV*
