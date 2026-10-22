# Contexto do Projeto - Previsão de Retorno para Revisão Honda

**Documento de Handoff para Continuidade**  
*Última atualização: 20/10/2026*

---

## 1. Visão Geral do Projeto

### Objetivo
Atividade de Avaliação da disciplina **Modelagem Preditiva** da **FGV**, com entrega até **22/10/2026**.

### Problema de Negócio
Prever se clientes que compraram motocicletas Honda irão retornar para a **primeira revisão programada** na concessionária.

### Tipo de Modelagem
**Classificação Binária**
- Classe 0: Cliente NÃO retornou para revisão
- Classe 1: Cliente RETORNOU para revisão

---

## 2. Grupo

| Nome | Matrícula |
|------|-----------|
| João Victor Fortunato Teixeira Mundim | 813190/2026 |
| Martinho Aloisio Lenz | 812869/2026 |
| Renato Vargas Monteiro | 812872/2026 |
| Rodrigo Pimentel Carneiro | 812440/2026 |

---

## 3. Dataset

### Arquivo Principal
`dados/Honda_Revisao_Modelagem.csv`

### Características
| Item | Valor |
|------|-------|
| Registros | 10.467 |
| Variáveis | 16 (15 preditoras + 1 alvo) |
| Separador | vírgula (,) |
| Encoding | UTF-8 |
| Distribuição alvo | 73,9% Não Retornou / 26,1% Retornou |

### Variáveis do Dataset
```
MOTO_MODELO, MOTO_FAMILIA, COR, ANO_MODELO, PRECO_TABELA, 
VALOR_VENDIDO, DESCONTO_INCONDICIONAL, PERC_DESCONTO, 
TEM_FINANCIAMENTO, FINANCEIRA_RESUMIDA, MESES, PARCELA, 
TEM_KIT, TEM_VOUCHER, ONDE_MOTO_VENDIDA, RETORNOU_REVISAO
```

### Origem dos Dados
Dados reais da concessionária Honda (anonimizados). Dados sensíveis removidos:
- CHASSI_COMPLETO (identificador)
- Números de notas fiscais
- Códigos internos
- Não havia CPF ou dados pessoais de clientes

---

## 4. Modelos Implementados

### 4.1 Árvore de Decisão (ID3/Entropy)
```python
DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
```

### 4.2 Regressão Logística
```python
LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
```
*Nota: Usa One-Hot Encoding + StandardScaler*

### 4.3 Random Forest
```python
RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=10,
    max_features='sqrt',
    random_state=42
)
```

### 4.4 Gradient Boosting
```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
```

---

## 5. Resultados Finais (Valores Reais)

| Modelo | Acurácia | AUC-ROC | Precisão | Recall | F1-Score | CV (5-fold) |
|--------|----------|---------|----------|--------|----------|-------------|
| **Random Forest** | **76,84%** | **0,8075** | **61,42%** | 30,04% | 40,34% | 58,35% ±14,74% |
| Regressão Logística | 76,74% | 0,8071 | 57,87% | 39,74% | 47,12% | 57,39% ±14,22% |
| Gradient Boosting | 76,31% | 0,8066 | 57,58% | 34,80% | 43,38% | 59,43% ±14,08% |
| Árvore de Decisão | 74,98% | 0,7546 | 53,96% | 27,47% | 36,41% | 60,41% ±12,16% |

### Melhor Modelo
**Random Forest** com AUC-ROC = 0,8075

### Features Mais Importantes
1. DESCONTO_INCONDICIONAL (0,1156)
2. VALOR_VENDIDO (0,1005)
3. PRECO_TABELA (0,0933)
4. PERC_DESCONTO (0,0915)
5. PARCELA (0,0791)
6. MOTO_MODELO (0,0744)
7. ONDE_MOTO_VENDIDA (0,0733)
8. TEM_KIT (0,0717)

---

## 6. Estrutura de Arquivos

```
atividade_avaliacao/
├── Relatorio_Honda_Revisao_FGV.md      # Relatório principal (Markdown)
├── Relatorio_Honda_Revisao_FGV.pdf     # Relatório em PDF (entregável)
├── README.md                            # Descrição do projeto
├── CONTEXTO_PROJETO.md                  # Este arquivo de handoff
├── dados/
│   ├── Honda_Revisao_Modelagem.csv     # Dataset principal
│   └── Revisao_Honda_Treinamento.csv   # Dataset original
├── notebooks/
│   └── Modelos_Honda_Revisao.ipynb     # Notebook com todos os modelos
├── resultados/
│   ├── graficos_comparativos.png       # ROC, métricas, confusão
│   └── matrizes_confusao.png           # Matrizes individuais
└── backup_churn/                        # Arquivos do tema anterior
```

---

## 7. Requisitos do Enunciado (Checklist)

- [x] Enunciado do problema (família de modelagem, contexto de negócio)
- [x] Engenharia de variáveis (seleção, encoding, transformações)
- [x] 4 modelos obrigatórios:
  - [x] Árvore de Decisão (Entropia/ID3)
  - [x] Regressão Logística
  - [x] Random Forest
  - [x] Gradient Boosting
- [x] Documentação de parâmetros testados
- [x] Quadro comparativo de desempenho
- [x] Resumo final com roteiro de produção
- [x] Relatório em PDF (máximo 6 páginas)
- [x] Notebook com código executável
- [x] Organização no GitHub

---

## 8. Comandos Úteis

### Executar os Modelos
```bash
cd atividade_avaliacao
pip install pandas numpy scikit-learn matplotlib seaborn
python3 -c "exec(open('notebooks/Modelos_Honda_Revisao.ipynb.py').read())"
```

### Gerar PDF do Relatório
```bash
pandoc Relatorio_Honda_Revisao_FGV.md -o Relatorio_Honda_Revisao_FGV.pdf \
    --pdf-engine=pdflatex -V geometry:margin=2cm -V fontsize=11pt
```

### Carregar Dataset
```python
import pandas as pd
df = pd.read_csv('dados/Honda_Revisao_Modelagem.csv')
```

---

## 9. Histórico de Decisões

| Data | Decisão | Motivo |
|------|---------|--------|
| 20/10/2026 | Tema mudado de "Churn" para "Revisão Honda" | Relevância para o trabalho do Rodrigo |
| 20/10/2026 | Removidos dados sensíveis (CHASSI, NFs) | Compliance e privacidade |
| 20/10/2026 | Simplificação de cores (46→7 categorias) | Melhor generalização |
| 20/10/2026 | Random Forest escolhido como melhor | Maior AUC-ROC (0,8075) |

---

## 10. Pendências e Melhorias Futuras

### Pendente
- [ ] Verificar número de páginas do PDF (máximo 6)
- [ ] Revisar formatação do PDF para entrega

### Melhorias Futuras (Pós-Entrega)
- Testar GridSearchCV para otimização de hiperparâmetros
- Implementar SMOTE para balanceamento de classes
- Adicionar XGBoost ou LightGBM para comparação
- Criar API de predição para uso em produção
- Dashboard de monitoramento com Streamlit

---

## 11. Repositório

**GitHub:** https://github.com/rodrigopimentel12/aula-predicao

**Branch principal:** main

---

## 12. Contato

Rodrigo Pimentel Carneiro (812440/2026)  
Concessionária Honda - Dados de vendas e revisões

---

*Este documento serve como ponte de contexto para continuidade do trabalho por outra IA ou desenvolvedor.*
