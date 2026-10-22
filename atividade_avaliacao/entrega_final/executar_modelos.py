#!/usr/bin/env python3
"""
Previsão de Retorno para Revisão - Concessionária Honda
FGV - Modelagem Preditiva - Entrega 22/10/2026

Script com pipeline corrigido (sem vazamento de dados).
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)


def carregar_dados(caminho=None):
    if caminho is None:
        from pathlib import Path
        caminho = Path(__file__).resolve().parent / 'Honda_Revisao_Modelagem.csv'
    """Carrega e prepara os dados."""
    df = pd.read_csv(caminho)
    print(f"Dataset carregado: {df.shape[0]} registros, {df.shape[1]} variáveis")
    print(f"Taxa de retorno: {df['RETORNOU_REVISAO'].mean()*100:.1f}%")
    print(f"Classe 0 (não retornou): {(df['RETORNOU_REVISAO']==0).sum()}")
    print(f"Classe 1 (retornou): {(df['RETORNOU_REVISAO']==1).sum()}")
    return df


def preparar_dados(df):
    """Separa features e target, identifica tipos."""
    X = df.drop(columns=['RETORNOU_REVISAO']).copy()
    y = df['RETORNOU_REVISAO'].copy()
    
    colunas_cat = X.select_dtypes(include=['object']).columns.tolist()
    colunas_num = X.select_dtypes(include=['number']).columns.tolist()
    
    print(f"\nVariáveis categóricas ({len(colunas_cat)}): {colunas_cat}")
    print(f"Variáveis numéricas ({len(colunas_num)}): {colunas_num}")
    
    return X, y, colunas_cat, colunas_num


def treinar_e_avaliar(X, y, colunas_cat, colunas_num):
    """Treina os 4 modelos com pipeline correto (sem vazamento)."""
    random_state = 42
    
    # CORREÇÃO: Dividir ANTES de qualquer transformação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    print(f"\nTreino: {len(y_train)} | Teste: {len(y_test)}")
    print(f"Baseline (classe majoritária): {(y_test==0).mean()*100:.2f}%")
    
    # Preparar encoders para modelos de árvore (fit apenas no treino)
    X_train_tree = X_train.copy()
    X_test_tree = X_test.copy()
    encoders = {}
    
    for col in colunas_cat:
        le = LabelEncoder()
        X_train_tree[col] = le.fit_transform(X_train[col].astype(str))
        # Para teste, tratar categorias não vistas
        X_test_tree[col] = X_test[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        encoders[col] = le
    
    # Preparar dados para Regressão Logística (fit apenas no treino)
    X_train_log = pd.get_dummies(X_train, columns=colunas_cat, drop_first=True)
    X_test_log = pd.get_dummies(X_test, columns=colunas_cat, drop_first=True)
    
    # Alinhar colunas (teste pode ter categorias diferentes)
    for col in X_train_log.columns:
        if col not in X_test_log.columns:
            X_test_log[col] = 0
    X_test_log = X_test_log[X_train_log.columns]
    
    # CORREÇÃO: Scaler fit apenas no treino
    scaler = StandardScaler()
    X_train_log_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_log), 
        columns=X_train_log.columns,
        index=X_train_log.index
    )
    X_test_log_scaled = pd.DataFrame(
        scaler.transform(X_test_log),
        columns=X_test_log.columns,
        index=X_test_log.index
    )
    
    resultados = {}
    
    # Modelo 1: Árvore de Decisão (CART com Entropia)
    print("\nTreinando Árvore de Decisão (CART/Entropia)...")
    arvore = DecisionTreeClassifier(
        criterion='entropy', max_depth=5, min_samples_split=20, 
        min_samples_leaf=10, random_state=random_state
    )
    arvore.fit(X_train_tree, y_train)
    y_pred_arvore = arvore.predict(X_test_tree)
    y_proba_arvore = arvore.predict_proba(X_test_tree)[:, 1]
    
    # CORREÇÃO: CV apenas no treino, com shuffle
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_arvore = cross_val_score(arvore, X_train_tree, y_train, cv=cv, scoring='accuracy')
    
    resultados['Árvore de Decisão'] = {
        'modelo': arvore,
        'y_pred': y_pred_arvore,
        'y_proba': y_proba_arvore,
        'cv_scores': cv_arvore
    }
    
    # Modelo 2: Regressão Logística
    print("Treinando Regressão Logística...")
    logistica = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=random_state
    )
    logistica.fit(X_train_log_scaled, y_train)
    y_pred_log = logistica.predict(X_test_log_scaled)
    y_proba_log = logistica.predict_proba(X_test_log_scaled)[:, 1]
    cv_log = cross_val_score(logistica, X_train_log_scaled, y_train, cv=cv, scoring='accuracy')
    
    resultados['Regressão Logística'] = {
        'modelo': logistica,
        'y_pred': y_pred_log,
        'y_proba': y_proba_log,
        'cv_scores': cv_log
    }
    
    # Modelo 3: Random Forest
    print("Treinando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, criterion='entropy', max_depth=10, 
        max_features='sqrt', random_state=random_state, n_jobs=-1
    )
    rf.fit(X_train_tree, y_train)
    y_pred_rf = rf.predict(X_test_tree)
    y_proba_rf = rf.predict_proba(X_test_tree)[:, 1]
    cv_rf = cross_val_score(rf, X_train_tree, y_train, cv=cv, scoring='accuracy')
    
    resultados['Random Forest'] = {
        'modelo': rf,
        'y_pred': y_pred_rf,
        'y_proba': y_proba_rf,
        'cv_scores': cv_rf,
        'feature_names': X_train_tree.columns.tolist()
    }
    
    # Modelo 4: Gradient Boosting
    print("Treinando Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, 
        subsample=0.8, random_state=random_state
    )
    gb.fit(X_train_tree, y_train)
    y_pred_gb = gb.predict(X_test_tree)
    y_proba_gb = gb.predict_proba(X_test_tree)[:, 1]
    cv_gb = cross_val_score(gb, X_train_tree, y_train, cv=cv, scoring='accuracy')
    
    resultados['Gradient Boosting'] = {
        'modelo': gb,
        'y_pred': y_pred_gb,
        'y_proba': y_proba_gb,
        'cv_scores': cv_gb
    }
    
    return resultados, y_test, X_train_tree.columns.tolist()


def exibir_resultados(resultados, y_test):
    """Exibe quadro comparativo de resultados."""
    print("\n" + "="*95)
    print("QUADRO COMPARATIVO DE RESULTADOS")
    print("="*95)
    
    baseline = (y_test == 0).mean()
    print(f"\nBaseline (sempre prever classe 0): {baseline*100:.2f}%")
    
    print(f"\n{'Modelo':<22} {'Acurácia':>10} {'AUC-ROC':>10} {'Prec(1)':>10} {'Rec(1)':>10} {'Rec(0)':>10} {'F1(1)':>10} {'CV(treino)':>16}")
    print("-"*106)
    
    tabela = []
    for nome, dados in resultados.items():
        y_pred = dados['y_pred']
        y_proba = dados['y_proba']
        cv = dados['cv_scores']
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)  # Recall classe 1 (retornou)
        rec0 = recall_score(y_test, y_pred, pos_label=0)  # Recall classe 0 (não retornou)
        f1 = f1_score(y_test, y_pred)
        cv_mean = cv.mean()
        cv_std = cv.std()
        
        print(f"{nome:<22} {acc*100:>9.2f}% {auc:>10.4f} {prec*100:>9.2f}% {rec*100:>9.2f}% {rec0*100:>9.2f}% {f1*100:>9.2f}% {cv_mean*100:>7.2f}% ±{cv_std*100:>5.2f}%")
        
        tabela.append({
            'Modelo': nome,
            'Acurácia': acc,
            'AUC-ROC': auc,
            'Precisão': prec,
            'Recall_1': rec,
            'Recall_0': rec0,
            'F1': f1,
            'CV_mean': cv_mean,
            'CV_std': cv_std
        })
    
    # Identificar melhor modelo
    aucs = [(dados['AUC-ROC'], dados['Modelo']) for dados in tabela]
    melhor_auc, melhor_nome = max(aucs, key=lambda x: x[0])
    
    print("\n" + "="*95)
    print(f"MELHOR MODELO (por AUC-ROC): {melhor_nome} (AUC = {melhor_auc:.4f})")
    print("="*95)
    
    return tabela, melhor_nome


def gerar_graficos(resultados, y_test, feature_names, output_dir='.'):
    """Gera gráficos comparativos."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Curvas ROC
    cores = {'Árvore de Decisão': 'blue', 'Regressão Logística': 'purple', 
             'Random Forest': 'orange', 'Gradient Boosting': 'green'}
    
    for nome, dados in resultados.items():
        y_proba = dados['y_proba']
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
    modelos = list(resultados.keys())
    metricas = {
        'Acurácia': [accuracy_score(y_test, resultados[m]['y_pred']) for m in modelos],
        'Precisão': [precision_score(y_test, resultados[m]['y_pred']) for m in modelos],
        'Recall': [recall_score(y_test, resultados[m]['y_pred']) for m in modelos],
        'F1-Score': [f1_score(y_test, resultados[m]['y_pred']) for m in modelos]
    }
    
    x = np.arange(len(modelos))
    width = 0.2
    
    for i, (metrica, valores) in enumerate(metricas.items()):
        axes[0,1].bar(x + i*width, valores, width, label=metrica)
    
    axes[0,1].set_xticks(x + width*1.5)
    axes[0,1].set_xticklabels(modelos, rotation=45, ha='right')
    axes[0,1].set_ylabel('Valor')
    axes[0,1].set_title('Comparação de Métricas por Modelo')
    axes[0,1].legend()
    axes[0,1].set_ylim(0, 1)
    
    # 3. Matrizes de Confusão (todos os modelos)
    # CORREÇÃO: Usar mapeamento correto nome->predição
    aucs = [(roc_auc_score(y_test, resultados[m]['y_proba']), m) for m in modelos]
    melhor_auc, melhor_nome = max(aucs, key=lambda x: x[0])
    melhor_pred = resultados[melhor_nome]['y_pred']
    
    cm = confusion_matrix(y_test, melhor_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0],
                xticklabels=['Não Retornou', 'Retornou'],
                yticklabels=['Não Retornou', 'Retornou'])
    axes[1,0].set_title(f'Matriz de Confusão - {melhor_nome}')
    axes[1,0].set_xlabel('Predito')
    axes[1,0].set_ylabel('Real')
    
    # 4. Importância das variáveis (Random Forest)
    rf_modelo = resultados['Random Forest']['modelo']
    importancias = pd.DataFrame({
        'Feature': feature_names,
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
    
    # Gerar matrizes individuais
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    axes2 = axes2.flatten()
    
    for i, (nome, dados) in enumerate(resultados.items()):
        cm = confusion_matrix(y_test, dados['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes2[i],
                    xticklabels=['Não Retornou', 'Retornou'],
                    yticklabels=['Não Retornou', 'Retornou'])
        axes2[i].set_title(f'{nome}')
        axes2[i].set_xlabel('Predito')
        axes2[i].set_ylabel('Real')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/matrizes_confusao.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráficos salvos em {output_dir}/")


def main():
    """Função principal."""
    print("="*70)
    print("PREVISÃO DE RETORNO PARA REVISÃO - HONDA")
    print("FGV - Modelagem Preditiva - Entrega 22/10/2026")
    print("Pipeline corrigido (sem vazamento de dados)")
    print("="*70)
    
    # Carregar dados
    df = carregar_dados()
    
    # Preparar dados
    X, y, colunas_cat, colunas_num = preparar_dados(df)
    
    # Treinar e avaliar modelos
    resultados, y_test, feature_names = treinar_e_avaliar(X, y, colunas_cat, colunas_num)
    
    # Exibir resultados
    tabela, melhor_modelo = exibir_resultados(resultados, y_test)
    
    # Gerar gráficos
    gerar_graficos(resultados, y_test, feature_names)
    
    print("\nExecução concluída!")
    return resultados, tabela, melhor_modelo


if __name__ == "__main__":
    main()
