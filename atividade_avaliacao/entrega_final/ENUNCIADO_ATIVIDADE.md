# Enunciado Oficial da Atividade de Avaliação
## Modelagem Preditiva - FGV

**Vencimento:** 22 de outubro de 2026 às 23:59

---

## Objetivo

Esta atividade vai permitir que você realmente trabalhe como um Cientista de Dados, ainda que em um contexto mais simples, exigindo preparação dos dados através de Engenharia de Variáveis (Feature Engineering), exploração de diversas modelagens (para comparações e refinamentos de seus desempenhos) e formulação final de um mecanismo de predição para algum problema ou oportunidade de negócio.

---

## Passo 1 - Identificar o Tema

Identifique, no contexto profissional onde você atua, um tema de negócio que possa se beneficiar de alguma análise preditiva.

**Exemplos:**
- Previsão de inadimplência de clientes (classificação)
- Previsão de vendas mensais (regressão)
- Previsão de rotatividade de funcionários (classificação)
- Previsão de tempo de atendimento (regressão)

---

## Passo 2 - Desenvolvimento

Para o tema escolhido:

1. **Enuncie claramente** o problema / oportunidade que se deseja tratar com dados.

2. **Identifique as fontes de dados** cabíveis para a análise.

3. **Determine a família de modelagem** analítica que será usada (classificação, regressão).

4. **Obtenha um conjunto de dados** para o treinamento (reais ou fictícios).

5. **Aplique técnicas de Engenharia de Variáveis** (Feature Engineering) sobre os dados.

6. **Prepare os modelos** que você vai executar (em Jupyter Notebooks ou outra ferramenta). Este conjunto de modelos deve **obrigatoriamente** conter, no mínimo:

   - **Se classificação:** uma árvore por entropia clássica (ID3) ou Gini + Regressão Logística + Random Forest + Gradient Boosting para classificação.
   
   - **Se regressão:** regressão linear ou polinomial + Gradient Boosting para regressão.

7. **Execute as diferentes modelagens**, sempre lançando mão de dados de retenção (holdout data), explorando os diferentes parâmetros de cada modelo e apurando indicadores de desempenho das diferentes versões que você executar (por exemplo: recall, precisão, acurácia, F1-score, posição em gráfico ROC). Não se esqueça de considerar o particular contexto de negócio em questão (que indicadores fazem mais sentido para ele).

8. **Apresente um resumo final** de como as predições deverão ser realizadas em produção (o que e como executar, como apresentar os resultados, que desempenho esperar).

---

## Passo 3 - Entregas

### Documento Principal (obrigatório)

Um texto em **MS Word ou PDF**, com **no máximo 06 (seis) páginas**, com os seguintes tópicos:

1. Enunciado do problema / oportunidade.

2. Explicação resumida de que técnicas de Engenharia de Variáveis (Feature Engineering) você utilizou para preparar os dados e por quê.

3. Que modelagens você executou.

4. Que tentativas você fez para melhorar o desempenho de cada uma das modelagens (resumidamente). Por exemplo, descreva as podas realizadas em árvores, as execuções de modelos ensemble com diferentes configurações de parâmetros para comparar os desempenhos etc.

5. Os resultados (prints) das execuções das modelagens (tabelas / gráficos de resultados) para alguns exemplos a predizer a classe ou um valor.

6. Um quadro de comparação de desempenho entre os modelos / versões de modelos utilizados.

7. Um resumo final comparando os resultados obtidos em cada modelagem / versão de modelagem e que roteiro deve ser utilizado em produção para a execução da predição (o que executar, como mostrar o resultado).

### Observação

**Não é obrigatório** entregar a planilha de dados de treinamento nem os códigos-fonte utilizados.

---

## Passo 4 - Formato do Trabalho

O trabalho pode ser feito **individualmente** ou em **grupos de até 5 pessoas**.

---

## Passo 5 - Identificação do Grupo

No caso de trabalho em grupo, identifique no documento Word da entrega as pessoas elaboradoras (e pode subir as entregas na plataforma no nome de apenas um dos membros do grupo).

---

## Passo 6 - Recomendações e Orientações Finais

- **Utilize a IA como uma ferramenta de auxílio**, onde cabível (para gerar e adaptar código, ajudar na interpretação), **não como a executora de todo o trabalho** apenas para você conferir e entregar. Procure explicar com as suas palavras as decisões de modelagem e os resultados obtidos.

- **O objetivo aqui não é você conseguir uma modelagem de predição maravilhosa**, mas sim demonstrar que manejou os conceitos e as técnicas que conhece rumo ao melhor resultado possível.

- Ou seja, **quem está em avaliação não são os modelos**, mas sim a capacidade de os(as) alunos(as) usarem de forma razoável os conceitos e técnicas vistas na disciplina.

- **Seja simples!**

---

*Disciplina: Modelagem Preditiva - FGV*
