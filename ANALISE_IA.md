# Análise e Ponderações - Modelagem Preditiva FGV

> **Grupo:**
> - João Victor Fortunato Teixeira Mundim (813190/2026)
> - Martinho Aloisio Lenz (812869/2026)
> - Nicolas Henrique Richardt da Silva (813138/2026)
> - Renato Vargas Monteiro (812872/2026)
> - Rodrigo Pimentel Carneiro (812440/2026)

---

## 📚 Visão Geral do Curso

O curso de **Modelagem Preditiva** (Prof. Roberto Pina Rizzo) possui **283 slides** organizados em 5 módulos:

| Módulo | Tema |
|--------|------|
| 1 | Ciclo de execução dos projetos analíticos |
| 2 | Recapitulação - modelagens preditivas de Ciência de Dados |
| 3 | Engenharia de Variáveis (Feature Engineering) |
| 4 | Construção e validação de modelagens |
| 5 | Consolidação de modelagens preditivas (Ensemble) |

---

## 📊 Análise do Dataset de Treinamento

### Estrutura dos Dados

| Característica | Valor |
|----------------|-------|
| Total de registros | 15 |
| Total de variáveis | 6 |
| Valores ausentes | 2 (na coluna Renda_Mensal) |

### Variáveis Disponíveis

| Variável | Tipo | Descrição |
|----------|------|-----------|
| `Idade` | Numérica (int) | Idade do cliente (22-72 anos) |
| `Renda_Mensal` | Numérica (float) | Renda mensal em R$ (2.200 - 30.000) |
| `Estado_Civil` | Categórica | Casado ou Solteiro |
| `Valor_Compras` | Numérica (int) | Valor total de compras (300 - 4.500) |
| `Numero_Compras` | Numérica (int) | Quantidade de compras (2 - 30) |
| `Classe` | Binária (target) | 0 = Cliente comum, 1 = Cliente excelente |

### Estatísticas Descritivas

| Variável | Média | Desvio Padrão | Mín | Máx |
|----------|-------|---------------|-----|-----|
| Idade | 48.07 | 15.27 | 22 | 72 |
| Renda_Mensal | 10.138,46 | 8.127,68 | 2.200 | 30.000 |
| Valor_Compras | 1.818,00 | 1.304,42 | 300 | 4.500 |
| Numero_Compras | 12.07 | 8.22 | 2 | 30 |

### Distribuição da Classe Alvo

- **Classe 0** (Cliente comum): 7 registros (46,7%)
- **Classe 1** (Cliente excelente): 8 registros (53,3%)

---

## 🔍 Ponderações e Observações

### 1. Sobre o Dataset

#### ⚠️ Pontos de Atenção:

1. **Tamanho reduzido**: Apenas 15 registros é insuficiente para modelos robustos. Em produção, precisaríamos de centenas ou milhares de registros.

2. **Valores ausentes**: Existem 2 valores `NaN` na coluna `Renda_Mensal` (linhas 7 e 12). Estratégias de tratamento:
   - Imputação pela mediana (recomendado para evitar viés de outliers)
   - Imputação pela média do grupo (por Classe)
   - Remoção das linhas (não recomendado dado o tamanho pequeno)

3. **Desbalanceamento baixo**: A proporção 46,7% vs 53,3% é razoavelmente equilibrada.

4. **Estado_Civil**: Baixa variabilidade - apenas 1 registro "Solteiro" entre 15. Considerar descarte.

### 2. Correlações Esperadas

Baseado na estrutura dos dados, espera-se:

- **Alta correlação** entre `Idade`, `Renda_Mensal`, `Valor_Compras` e `Numero_Compras`
- Clientes mais velhos tendem a ter maior renda e mais compras
- Possível candidata a **PCA** (Principal Component Analysis)

### 3. Feature Engineering Sugerido

| Técnica | Aplicação Sugerida |
|---------|-------------------|
| **Imputação** | Usar mediana para `Renda_Mensal` ausente |
| **Descarte** | Considerar remover `Estado_Civil` (baixa variabilidade) |
| **Normalização** | Aplicar em `Renda_Mensal` (range muito amplo: 2.200-30.000) |
| **Binning** | Criar faixas de idade (Jovem, Adulto, Sênior) |
| **One-Hot Encoding** | Se manter `Estado_Civil`, converter para colunas binárias |
| **Recombinação** | Criar variável `Ticket_Medio = Valor_Compras / Numero_Compras` |

### 4. Modelos Preditivos Aplicáveis

Para classificação binária (Cliente comum vs excelente):

| Modelo | Adequação | Observação |
|--------|-----------|------------|
| Regressão Logística | ✅ Boa | Bom para baseline e interpretabilidade |
| Árvore de Decisão | ✅ Boa | Fácil interpretação, cuidado com overfitting |
| Random Forest | ⚠️ Média | Dataset muito pequeno para ensemble |
| SVM | ⚠️ Média | Pode funcionar, mas precisa tuning |
| KNN | ⚠️ Média | Sensível a escala, normalizar antes |

### 5. Métricas de Avaliação Sugeridas

- **Acurácia**: Proporção de acertos (válida pois classes são balanceadas)
- **Precisão/Recall**: Para entender falsos positivos/negativos
- **F1-Score**: Média harmônica entre precisão e recall
- **AUC-ROC**: Capacidade discriminativa do modelo
- **Matriz de Confusão**: Visualização dos erros

### 6. Validação

Com apenas 15 registros:
- **Leave-One-Out Cross-Validation (LOOCV)**: Mais apropriado
- **K-Fold com k=3 ou k=5**: Alternativa viável
- ⚠️ Evitar split simples train/test (poucos dados)

---

## 📝 Exercícios - Pontos-Chave

O documento de atividades foca em **Feature Engineering** com 9 questões:

1. **Seleção de variáveis**: Avaliar relevância de cada feature
2. **Correlação**: Usar `=CORREL()` no Excel para identificar multicolinearidade
3. **Recombinação**: Criar novas variáveis a partir das existentes
4. **Descarte**: Eliminar variáveis redundantes ou irrelevantes
5. **Valores ausentes**: Tratar os NaN em Renda_Mensal
6. **Transformação**: Log, normalização ou binning para assimetria
7. **One-Hot Encoding**: Converter Estado_Civil em variáveis dummy
8. **Poder discriminatório**: Testar ANOVA e Qui-Quadrado
9. **PCA**: Redução de dimensionalidade

---

## 🛠️ Código Python de Apoio

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Carregar dados
df = pd.read_excel('dados/Dados de Treinamento.xlsx')

# Tratar valores ausentes (mediana)
df['Renda_Mensal'].fillna(df['Renda_Mensal'].median(), inplace=True)

# Criar variável derivada
df['Ticket_Medio'] = df['Valor_Compras'] / df['Numero_Compras']

# Verificar correlações
print(df.corr())

# Preparar features e target
X = df[['Idade', 'Renda_Mensal', 'Valor_Compras', 'Numero_Compras']]
y = df['Classe']

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelo com LOOCV
loo = LeaveOneOut()
model = LogisticRegression()
scores = cross_val_score(model, X_scaled, y, cv=loo)
print(f'Acurácia média (LOOCV): {scores.mean():.2%}')
```

---

## 📎 Links Úteis

- [Scikit-learn - Documentação](https://scikit-learn.org/stable/)
- [Pandas - Documentação](https://pandas.pydata.org/docs/)
- [Seaborn - Visualização](https://seaborn.pydata.org/)

---

## 🤖 Notas para Continuidade por Outras IAs

Este documento foi criado para permitir que outras IAs continuem o trabalho. Informações importantes:

1. **Arquivos no repositório**:
   - `aulas/Modelagem Preditiva.pdf` - 283 slides do curso
   - `dados/Dados de Treinamento.xlsx` - Dataset com 15 registros e 6 colunas
   - `exercicios/Atividades - Modelagem Preditiva.docx` - 9 questões de Feature Engineering

2. **Próximos passos sugeridos**:
   - Resolver cada uma das 9 questões do exercício
   - Criar notebook Jupyter com análise exploratória completa
   - Implementar e comparar diferentes modelos de classificação
   - Documentar resultados no repositório

3. **Dependências Python necessárias**:
   ```
   pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl
   ```

---

*Análise gerada automaticamente em 26/09/2026*
