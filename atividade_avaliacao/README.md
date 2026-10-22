# Atividade de Avaliação - Modelagem Preditiva

**FGV - Modelagem Preditiva**

**Data de entrega:** 22 de outubro de 2026

**Grupo:**
- João Victor Fortunato Teixeira Mundim (813190/2026)
- Rodrigo Pimentel Carneiro (812440/2026)
- Bruno Barradas

---

## 🏍️ Tema Escolhido: Previsão de Retorno para Revisão - Honda

### Objetivo
Prever se um cliente que comprou uma motocicleta Honda irá retornar para a **primeira revisão programada**.

### Problema de Negócio
- **73,9%** dos clientes NÃO retornam para a revisão
- Identificar antecipadamente quem não vai retornar permite ações proativas
- Aumentar a taxa de retorno gera receita e fidelização

---

## Estrutura da Pasta

```
atividade_avaliacao/
│
├── 📄 Relatorio_Honda_Revisao_FGV.md    ← DOCUMENTO PARA ENTREGA
│
├── dados/
│   ├── Honda_Revisao_Modelagem.csv      ← Dataset principal (10.467 registros)
│   └── Revisao_Honda_Treinamento.csv    ← Dataset intermediário
│
└── notebooks/
    └── Modelos_Honda_Revisao.ipynb      ← Notebook com os 4 modelos
```

---

## Dataset

| Característica | Valor |
|----------------|-------|
| **Fonte** | Dados reais da concessionária Honda (anonimizados) |
| **Registros** | 10.467 vendas |
| **Variáveis** | 16 (15 preditoras + 1 alvo) |
| **Período** | 2025 |

### Variável Alvo
| Classe | Quantidade | Percentual |
|--------|------------|------------|
| 0 - Não Retornou | 7.738 | 73,9% |
| 1 - Retornou | 2.729 | 26,1% |

### Variáveis Preditoras
- **Moto:** modelo, família, cor, ano
- **Valores:** preço tabela, valor vendido, desconto
- **Financiamento:** tem financiamento, banco, prazo, parcela
- **Acessórios:** kit, voucher
- **Local:** cidade da venda

---

## Modelos Implementados

| # | Modelo | Critério |
|---|--------|----------|
| 1 | Árvore de Decisão | Entropia (ID3) |
| 2 | Regressão Logística | L2 (Ridge) |
| 3 | Random Forest | Entropia |
| 4 | Gradient Boosting | Deviance |

---

## Como Executar

### Google Colab
1. Faça upload de `Honda_Revisao_Modelagem.csv` para o Drive
2. Abra `Modelos_Honda_Revisao.ipynb` no Colab
3. Ajuste o caminho do arquivo
4. Execute todas as células

### Localmente
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook Modelos_Honda_Revisao.ipynb
```

---

## Resultados Esperados

| Modelo | Acurácia | AUC-ROC |
|--------|----------|---------|
| Gradient Boosting | ~77% | ~0,74 |
| Random Forest | ~77% | ~0,74 |
| Regressão Logística | ~76% | ~0,72 |
| Árvore de Decisão | ~74% | ~0,68 |

---

**Professor:** Roberto Pina Rizzo
