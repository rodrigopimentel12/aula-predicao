#!/usr/bin/env python3
"""
Previsão de Retorno para Revisão - Concessionária Honda
FGV - Modelagem Preditiva

Script standalone para executar os 4 modelos de classificação.
Uso: python3 executar_modelos.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)


def carregar_dados(caminho='dados/Honda_Revisao_Modelagem.csv'):
    """Carrega e prepara os dados."""
    df = pd.read_csv(caminho)
    print(f"Dataset carregado: {df.shape[0]} registros, {df.shape[1]} variáveis")
    print(f"Taxa de retorno: {df['RETORNOU_REVISAO'].mean()*100:.1f}%")
    return df


def preparar_features(df):
    """Aplica feature engineering."""
    X = df.drop(columns=['RETORNOU_REVISAO']).copy()
    y = df['RETORNOU_REVISAO'].copy()
    
    colunas_cat = X.select_dtypes(include=['object']).columns.tolist()
    
    # Label Encoding para modelos de árvore
    X_tree = X.copy()
    encoders = {}
    for col in colunas_cat:
        le = LabelEncoder()
        X_tree[col] = le.fit_transform(X_tree[col].astype(str))
        encoders[col] = le
    
    # One-Hot Encoding para Regressão Logística
    X_log = pd.get_dummies(X, columns=colunas_cat, drop_first=True)
    scaler = StandardScaler()
    X_log_scaled = pd.DataFrame(scaler.fit_transform(X_log), columns=X_log.columns)
    
    return X_tree, X_log_scaled, y, encoders


def treinar_modelos(X_tree, X_log_scaled, y):
    """Treina os 4 modelos e retorna resultados."""
    random_state = 42
    
    # Divisão treino/teste
    X_tree_train, X_tree_test, y_train, y_test = train_test_split(
        X_tree, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_log_train, X_log_test, _, _ = train_test_split(
        X_log_scaled, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    print(f"\nTreino: {len(y_train)} | Teste: {len(y_test)}")
    
    # Modelo 1: Árvore de Decisão
    print("\nTreinando Árvore de Decisão...")
    arvore = DecisionTreeClassifier(
        criterion='entropy', max_depth=5, min_samples_split=20, 
        min_samples_leaf=10, random_state=random_state
    )
    arvore.fit(X_tree_train, y_train)
    y_pred_arvore = arvore.predict(X_tree_test)
    y_proba_arvore = arvore.predict_proba(X_tree_test)[:, 1]
    cv_arvore = cross_val_score(arvore, X_tree, y, cv=5, scoring='accuracy')
    
    # Modelo 2: Regressão Logística
    print("Treinando Regressão Logística...")
    logistica = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=random_state
    )
    logistica.fit(X_log_train, y_train)
    y_pred_log = logistica.predict(X_log_test)
    y_proba_log = logistica.predict_proba(X_log_test)[:, 1]
    cv_log = cross_val_score(logistica, X_log_scaled, y, cv=5, scoring='accuracy')
    
    # Modelo 3: Random Forest
    print("Treinando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, criterion='entropy', max_depth=10, 
        max_features='sqrt', random_state=random_state, n_jobs=-1
    )
    rf.fit(X_tree_train, y_train)
    y_pred_rf = rf.predict(X_tree_test)
    y_proba_rf = rf.predict_proba(X_tree_test)[:, 1]
    cv_rf = cross_val_score(rf, X_tree, y, cv=5, scoring='accuracy')
    
    # Modelo 4: Gradient Boosting
    print("Treinando Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, 
        subsample=0.8, random_state=random_state
    )
    gb.fit(X_tree_train, y_train)
    y_pred_gb = gb.predict(X_tree_test)
    y_proba_gb = gb.predict_proba(X_tree_test)[:, 1]
    cv_gb = cross_val_score(gb, X_tree, y, cv=5, scoring='accuracy')
    
    resultados = {
        'Árvore de Decisão': (y_pred_arvore, y_proba_arvore, cv_arvore, arvore),
        'Regressão Logística': (y_pred_log, y_proba_log, cv_log, logistica),
        'Random Forest': (y_pred_rf, y_proba_rf, cv_rf, rf),
        'Gradient Boosting': (y_pred_gb, y_proba_gb, cv_gb, gb)
    }
    
    return resultados, y_test, X_tree


def exibir_resultados(resultados, y_test):
    """Exibe quadro comparativo de resultados."""
    print("\n" + "="*90)
    print("QUADRO COMPARATIVO DE RESULTADOS")
    print("="*90)
    
    print(f"\n{'Modelo':<22} {'Acurácia':>10} {'AUC-ROC':>10} {'Precisão':>10} {'Recall':>10} {'F1-Score':>10} {'CV 5-fold':>18}")
    print("-"*92)
    
    for nome, (y_pred, y_proba, cv, modelo) in resultados.items():
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cv_mean = cv.mean()
        cv_std = cv.std()
        print(f"{nome:<22} {acc*100:>9.2f}% {auc:>10.4f} {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}% {cv_mean*100:>8.2f}% ±{cv_std*100:>5.2f}%")
    
    # Identificar melhor modelo
    aucs = [(roc_auc_score(y_test, proba), nome) for nome, (_, proba, _, _) in resultados.items()]
    melhor_auc, melhor_nome = max(aucs, key=lambda x: x[0])
    
    print("\n" + "="*90)
    print(f"MELHOR MODELO (por AUC-ROC): {melhor_nome} (AUC = {melhor_auc:.4f})")
    print("="*90)
    
    return melhor_nome


def gerar_graficos(resultados, y_test, X_tree, output_dir='resultados'):
    """Gera gráficos comparativos."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Curvas ROC
    cores = {'Árvore de Decisão': 'blue', 'Regressão Logística': 'purple', 
             'Random Forest': 'orange', 'Gradient Boosting': 'green'}
    
    for nome, (y_pred, y_proba, cv, modelo) in resultados.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        axes[0,0].plot(fpr, tpr, color=cores[nome], lw=2, label=f'{nome} (AUC={auc:.3f})')
    
    axes[0,0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0,0].set_xlabel('Taxa de Falso Positivo')
    axes[0,0].set_ylabel('Taxa de Verdadeiro Positivo')
    axes[0,0].set_title('Curvas ROC Comparativas')
    axes[0,0].legend(loc='lower right')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Comparação de métricas
    metricas_data = []
    for nome, (y_pred, y_proba, cv, modelo) in resultados.items():
        metricas_data.append({
            'Modelo': nome,
            'Acurácia': accuracy_score(y_test, y_pred),
            'Precisão': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        })
    
    df_metricas = pd.DataFrame(metricas_data)
    x = np.arange(len(df_metricas))
    width = 0.2
    
    for i, metrica in enumerate(['Acurácia', 'Precisão', 'Recall', 'F1-Score']):
        axes[0,1].bar(x + i*width, df_metricas[metrica], width, label=metrica)
    
    axes[0,1].set_xticks(x + width*1.5)
    axes[0,1].set_xticklabels(df_metricas['Modelo'], rotation=45, ha='right')
    axes[0,1].set_ylabel('Valor')
    axes[0,1].set_title('Comparação de Métricas por Modelo')
    axes[0,1].legend()
    axes[0,1].set_ylim(0, 1)
    
    # 3. Matriz de Confusão (melhor modelo)
    aucs = [(roc_auc_score(y_test, proba), nome, pred) for nome, (pred, proba, _, _) in resultados.items()]
    melhor_auc, melhor_nome, melhor_pred = max(aucs, key=lambda x: x[0])
    
    cm = confusion_matrix(y_test, melhor_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0],
                xticklabels=['Não Retornou', 'Retornou'],
                yticklabels=['Não Retornou', 'Retornou'])
    axes[1,0].set_title(f'Matriz de Confusão - {melhor_nome}')
    axes[1,0].set_xlabel('Predito')
    axes[1,0].set_ylabel('Real')
    
    # 4. Importância das variáveis (Random Forest)
    rf_modelo = resultados['Random Forest'][3]
    importancias = pd.DataFrame({
        'Feature': X_tree.columns,
        'Importância': rf_modelo.feature_importances_
    }).sort_values('Importância', ascending=True).tail(10)
    
    axes[1,1].barh(range(len(importancias)), importancias['Importância'], color='steelblue')
    axes[1,1].set_yticks(range(len(importancias)))
    axes[1,1].set_yticklabels(importancias['Feature'])
    axes[1,1].set_xlabel('Importância')
    axes[1,1].set_title('Top 10 Features Mais Importantes (Random Forest)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/graficos_comparativos.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráficos salvos em {output_dir}/graficos_comparativos.png")


def main():
    """Função principal."""
    print("="*70)
    print("PREVISÃO DE RETORNO PARA REVISÃO - HONDA")
    print("FGV - Modelagem Preditiva")
    print("="*70)
    
    # Carregar dados
    df = carregar_dados()
    
    # Preparar features
    X_tree, X_log_scaled, y, encoders = preparar_features(df)
    
    # Treinar modelos
    resultados, y_test, X_tree = treinar_modelos(X_tree, X_log_scaled, y)
    
    # Exibir resultados
    melhor_modelo = exibir_resultados(resultados, y_test)
    
    # Gerar gráficos
    gerar_graficos(resultados, y_test, X_tree)
    
    print("\nExecução concluída!")
    return resultados, melhor_modelo


if __name__ == "__main__":
    main()
