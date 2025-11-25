# -*- coding: utf-8 -*-
"""
Versão melhorada do script SVM:
- pruning aplicado globalmente (igual para treino e teste)
- matrizes de confusão normalizadas em porcentagem por linha
- cálculo de média ± desvio padrão elemento-a-elemento das MCs
- prints no terminal em formato LaTeX-like conforme solicitado
- geração de relatório HTML com imagens das MCs (em %)
"""
from libsvm.svmutil import svm_read_problem, svm_train, svm_predict
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import sys
from time import perf_counter as _timer
from jinja2 import Template
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ---------------------------
# Helpers de pruning e IO
# ---------------------------
def compute_selected_indices(y, x, threshold):
    """
    Computa índices de features selecionadas com base na correlação (pearson)
    Retorna: selected_indices (0-based), correlations array (same length as max_feature)
    """
    if not x or not y:
        return np.array([], dtype=int), np.array([])

    # identifica o maior índice de feature (LIBSVM usa 1-based)
    max_feature = 0
    for sample in x:
        if sample:
            max_feature = max(max_feature, max(sample.keys()))
    if max_feature == 0:
        return np.array([], dtype=int), np.array([])

    # montando matriz densa
    x_array = np.zeros((len(x), max_feature), dtype=float)
    for i, sample in enumerate(x):
        for feature, value in sample.items():
            x_array[i, feature - 1] = value

    correlations = np.zeros(x_array.shape[1], dtype=float)
    for i in range(x_array.shape[1]):
        col = x_array[:, i]
        if np.std(col) > 0:
            corr = np.corrcoef(col, y)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            correlations[i] = corr
        else:
            correlations[i] = 0.0

    selected_indices = np.where(np.abs(correlations) >= threshold)[0]
    return selected_indices, correlations

def apply_pruning_to_samples(x, selected_indices):
    """
    Aplica a seleção (selected_indices: 0-based) aos samples (lista de dicts)
    Retorna lista de samples com novos índices 1.. (LIBSVM style, mas nós retornamos dict)
    """
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices, start=1)}
    x_pruned = []
    for sample in x:
        pruned_sample = {}
        for feat, val in sample.items():
            zero_based = feat - 1
            if zero_based in mapping:
                pruned_sample[mapping[zero_based]] = val
        x_pruned.append(pruned_sample)
    return x_pruned, mapping

def save_libsvm(y, x, filename):
    """Salva os dados no formato LIBSVM (labels e dicts de features)."""
    with open(filename, 'w', encoding='utf-8') as f:
        for label, features in zip(y, x):
            sorted_features = sorted(features.items(), key=lambda item: item[0])
            feature_str = ' '.join(f"{k}:{v}" for k, v in sorted_features)
            f.write(f"{int(label)} {feature_str}\n")

# ---------------------------
# Plot de matriz de confusão (em %)
# ---------------------------
def plot_and_save_cm(cm_percent, title, filename, class_names=None):
    """Gera um heatmap a partir da matriz de confusão em porcentagem (valores floatt)"""
    if cm_percent is None:
        return
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap=plt.cm.Blues,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# ---------------------------
# Função principal de K-Fold
# ---------------------------
def svmKfold(y, x, t, cost, gamma, k=10):
    """
    Executa K-Fold usando os dados já prunados (x, y).
    Retorna:
      mean_train, std_train, mean_test, std_test,
      cm_mean_percent (2D array), cm_std_percent (2D array),
      mean_train_time, std_train_time, mean_test_time, std_test_time,
      unique_labels (list)
    """
    np.random.seed(1)
    kf = KFold(n_splits=k, shuffle=True)
    accuracies_train, accuracies_test = [], []
    cms_percent = []  # lista de matrizes de confusão por fold (em %)
    train_times, test_times = [], []

    unique_labels = sorted(np.unique(y))

    for train_index, test_index in kf.split(x):
        x_train = [x[i] for i in train_index]
        x_test  = [x[i] for i in test_index]
        y_train = [y[i] for i in train_index]
        y_test  = [y[i] for i in test_index]

        param_str = f'-t {t} -c {cost} -g {gamma} -q'

        # treino
        t0 = _timer()
        m = svm_train(y_train, x_train, param_str)
        train_times.append(_timer() - t0)

        # métricas de treino
        _, p_acc_train, _ = svm_predict(y_train, x_train, m)

        # teste (inferencia)
        t1 = _timer()
        p_label_test, p_acc_test, _ = svm_predict(y_test, x_test, m)
        test_times.append(_timer() - t1)

        accuracies_train.append(p_acc_train[0])
        accuracies_test.append(p_acc_test[0])

        # confusion matrix absoluta
        cm_abs = confusion_matrix(y_test, p_label_test, labels=unique_labels)
        # normalizar por linha (rótulo verdadeiro) -> porcentagem
        cm_percent = np.zeros_like(cm_abs, dtype=float)
        for i in range(cm_abs.shape[0]):
            row_sum = cm_abs[i].sum()
            if row_sum > 0:
                cm_percent[i, :] = (cm_abs[i, :] / row_sum) * 100.0
            else:
                cm_percent[i, :] = 0.0
        cms_percent.append(cm_percent)

    # converte lista para array 3D: (folds, n_classes, n_classes)
    cms_stack = np.stack(cms_percent, axis=0)  # shape: (k, n, n)
    cm_mean = np.mean(cms_stack, axis=0)
    cm_std  = np.std(cms_stack, axis=0)

    return (
        float(np.mean(accuracies_train)), float(np.std(accuracies_train)),
        float(np.mean(accuracies_test)), float(np.std(accuracies_test)),
        cm_mean, cm_std,
        float(np.mean(train_times)), float(np.std(train_times)),
        float(np.mean(test_times)), float(np.std(test_times)),
        unique_labels
    )

# ---------------------------
# Classe controladora do experimento
# ---------------------------
class svmParameters():
    def main(self, dataset, threshold):
        """Executa a busca de parâmetros e retorna resultados estruturados."""
        y, x = svm_read_problem(dataset)

        # --------------------------------------------------
        # PRUNING GLOBAL (aplicado para treino e teste)
        # --------------------------------------------------
        selected_indices, correlations = compute_selected_indices(y, x, threshold)
        if selected_indices.size == 0:
            print("Aviso: nenhuma feature selecionada com o threshold fornecido. Usando todas as features.")
            x_pruned = x
        else:
            x_pruned, mapping = apply_pruning_to_samples(x, selected_indices)
            # salva CSV e LIBSVM prunado
            with open('selected_features.csv', 'w', encoding='utf-8') as ff:
                ff.write("Original_Index,Selected_Index,Correlation\n")
                for new_idx, old_idx in enumerate(selected_indices, start=1):
                    ff.write(f"{old_idx+1},{new_idx},{correlations[old_idx]:.6f}\n")
            save_libsvm(y, x_pruned, 'globalPruned.csv')

        # vectores de busca (exemplo)
        cost_vector = [10**0, 10**3]
        gamma_vector = [10**0]

        # inicializa variáveis globais
        max_acc, min_acc = -1.0, 101.0
        max_cm_mean = None
        max_cm_std  = None
        min_cm_mean = None
        min_cm_std  = None

        max_kernel = min_kernel = -1
        max_cost = min_cost = 0
        max_gamma = min_gamma = 0

        max_train_mean = max_train_std = max_test_mean = max_test_std = 0.0
        min_train_mean = min_train_std = min_test_mean = min_test_std = 0.0

        max_mean_accuracy_train = max_std_accuracy_train = -1
        max_mean_accuracy_test  = max_std_accuracy_test = -1
        min_mean_accuracy_train = min_std_accuracy_train = 101
        min_mean_accuracy_test  = min_std_accuracy_test = 101

        kernel_results = {}

        for t in range(4):
            kernel_max_acc, kernel_min_acc = -1.0, 101.0
            kernel_max_data = {}
            kernel_min_data = {}

            print(f'\n=== Testando Kernel: {kernel_str(t)} ===')
            for c in cost_vector:
                for g in gamma_vector:
                    (mean_train, std_train, mean_test, std_test,
                     cm_mean, cm_std,
                     mean_train_time, std_train_time,
                     mean_test_time, std_test_time,
                     unique_labels) = svmKfold(y, x_pruned, t, c, g)

                    # armazenar resultado do kernel (melhor/pior para esse kernel)
                    if mean_test > kernel_max_acc:
                        kernel_max_acc = mean_test
                        kernel_max_data = {
                            "accuracy_train": mean_train, "std_train": std_train,
                            "accuracy_test": mean_test,  "std_test": std_test,
                            "confusion_mean": cm_mean, "confusion_std": cm_std,
                            "cost": c, "gamma": g,
                            "train_time_mean": mean_train_time, "train_time_std": std_train_time,
                            "test_time_mean": mean_test_time,  "test_time_std": std_test_time,
                            "labels": unique_labels
                        }
                    if mean_test < kernel_min_acc:
                        kernel_min_acc = mean_test
                        kernel_min_data = {
                            "accuracy_train": mean_train, "std_train": std_train,
                            "accuracy_test": mean_test,  "std_test": std_test,
                            "confusion_mean": cm_mean, "confusion_std": cm_std,
                            "cost": c, "gamma": g,
                            "train_time_mean": mean_train_time, "train_time_std": std_train_time,
                            "test_time_mean": mean_test_time,  "test_time_std": std_test_time,
                            "labels": unique_labels
                        }

                    # atualizar global best
                    if mean_test > max_acc:
                        max_acc = mean_test
                        max_kernel, max_cost, max_gamma = t, c, g
                        max_mean_accuracy_train, max_std_accuracy_train = mean_train, std_train
                        max_mean_accuracy_test,  max_std_accuracy_test  = mean_test, std_test
                        max_cm_mean, max_cm_std = cm_mean, cm_std
                        max_train_mean, max_train_std = mean_train_time, std_train_time
                        max_test_mean,  max_test_std  = mean_test_time, std_test_time

                    # atualizar global worst
                    if mean_test < min_acc:
                        min_acc = mean_test
                        min_kernel, min_cost, min_gamma = t, c, g
                        min_mean_accuracy_train, min_std_accuracy_train = mean_train, std_train
                        min_mean_accuracy_test,  min_std_accuracy_test  = mean_test, std_test
                        min_cm_mean, min_cm_std = cm_mean, cm_std
                        min_train_mean, min_train_std = mean_train_time, std_train_time
                        min_test_mean,  min_test_std  = mean_test_time, std_test_time

            kernel_results[t] = {"max": kernel_max_data, "min": kernel_min_data}

        # estrutura de resultados globais
        global_results = {
            "max": {"accuracy_train": max_mean_accuracy_train, "std_train": max_std_accuracy_train,
                    "accuracy_test":  max_mean_accuracy_test,  "std_test":  max_std_accuracy_test,
                    "kernel_id": max_kernel, "cost": max_cost, "gamma": max_gamma,
                    "confusion_mean": max_cm_mean, "confusion_std": max_cm_std,
                    "train_time_mean": max_train_mean, "train_time_std": max_train_std,
                    "test_time_mean":  max_test_mean,  "test_time_std":  max_test_std,
                    "labels": unique_labels},
            "min": {"accuracy_train": min_mean_accuracy_train, "std_train": min_std_accuracy_train,
                    "accuracy_test":  min_mean_accuracy_test,  "std_test":  min_std_accuracy_test,
                    "kernel_id": min_kernel, "cost": min_cost, "gamma": min_gamma,
                    "confusion_mean": min_cm_mean, "confusion_std": min_cm_std,
                    "train_time_mean": min_train_mean, "train_time_std": min_train_std,
                    "test_time_mean":  min_test_mean,  "test_time_std":  min_test_std,
                    "labels": unique_labels}
        }

        # prints detalhados no terminal (LaTeX-like) conforme solicitado
        self.print_terminal_summary(global_results, kernel_results)

        return global_results, kernel_results

    def print_terminal_summary(self, global_results, kernel_results):
        """Imprime no terminal:
           - resumo melhor/pior (formato com ±)
           - matrizes de confusão (melhor vs pior) no layout pedido
        """
        def fmt_mean_std(mean, std):
            return f"{mean:.2f} ± {std:.2f}"

        # MELHOR / PIOR (global)
        best = global_results['max']
        worst = global_results['min']
        best_kernel_name = kernel_str(best['kernel_id'])
        worst_kernel_name = kernel_str(worst['kernel_id'])
        # formatar C como 10^n se for potencia de 10
        def format_cost(c):
            if c > 0 and (math.log10(c)).is_integer():
                exp = int(round(math.log10(c)))
                return f"$10^{exp}$"
            else:
                return f"{c}"

        print("\n==========================================")
        print("RESUMO (formatado para o HTML / LaTeX-like)")
        print("==========================================")
        print(f"& Melhor: {best_kernel_name} & {format_cost(best['cost'])} & "
              f"{fmt_mean_std(best['accuracy_train'], best['std_train'])} & "
              f"{fmt_mean_std(best['accuracy_test'], best['std_test'])} & "
              f"{fmt_mean_std(best['train_time_mean'], best['train_time_std'])} & "
              f"{fmt_mean_std(best['test_time_mean'], best['test_time_std'])}\\\\")
        print(f"& Pior: {worst_kernel_name} & {format_cost(worst['cost'])} & "
              f"{fmt_mean_std(worst['accuracy_train'], worst['std_train'])} & "
              f"{fmt_mean_std(worst['accuracy_test'], worst['std_test'])} & "
              f"{fmt_mean_std(worst['train_time_mean'], worst['train_time_std'])} & "
              f"{fmt_mean_std(worst['test_time_mean'], worst['test_time_std'])}\\\\")
        print("==========================================\n")

        # MATRIZES DE CONFUSÃO: exibindo MELHOR e PIOR lado a lado, com média ± std por elemento
        bm = best['confusion_mean']
        bs = best['confusion_std']
        wm = worst['confusion_mean']
        ws = worst['confusion_std']
        labels = best.get('labels', None)

        # assumimos matriz quadrada n_classes x n_classes
        if bm is None or wm is None:
            print("Aviso: matriz de confusão melhor/pior não disponível.")
            return

        n = bm.shape[0]
        # imprimimos por linha (cada linha corresponde a um rótulo verdadeiro)
        # formato pedido: colocar elementos do melhor e do pior na mesma linha
        print("MATRIZES DE CONFUSÃO (porcentagem) — formato LaTeX-like\n")
        for i in range(n):
            row_label = labels[i] if labels is not None else f"R{i}"
            # melhor row: bm[i, 0..n-1] com bs
            best_elems = [f"{bm[i,j]:.2f} ± {bs[i,j]:.2f}" if (i==j) else f"{bm[i,j]:.2f} ± {bs[i,j]:.2f}" for j in range(n)]
            worst_elems = [f"{wm[i,j]:.2f} ± {ws[i,j]:.2f}" if (i==j) else f"{wm[i,j]:.2f} ± {ws[i,j]:.2f}" for j in range(n)]
            # construir a linha no formato desejado:
            # & <label> & <best row elements...> & <worst row elements...> \\
            # Para o seu exemplo (2 classes) isso resulta em 4 números ao todo depois do label.
            line_parts = [f" & {row_label} & "]
            line_parts.append(" & ".join(best_elems))
            line_parts.append(" & ")
            line_parts.append(" & ".join(worst_elems))
            line = "".join(line_parts) + " \\\\"
            print(line)
        print("\n(As células são mostradas como: média ± desvio padrão — porcentagem por linha)\n")

# ---------------------------
# util: nome do kernel
# ---------------------------
def kernel_str(t):
    if (t == 0):
        return 'Linear'
    elif (t == 1):
        return 'Polynomial'
    elif (t == 2):
        return 'Radial_Basis_Function'
    elif (t == 3):
        return 'Sigmoid'
    return 'Unknown'

# ---------------------------
# Geração do relatório HTML
# ---------------------------
def generate_html_report(global_results, kernel_results, output_file='svm_report.html'):
    """Gera relatório HTML e salva imagens das MCs (em porcentagem)."""
    img_dir = 'svm_report_images'
    os.makedirs(img_dir, exist_ok=True)

    # salva imagens das MCs (melhor / pior)
    best = global_results['max']
    worst = global_results['min']
    labels = best.get('labels', None)
    plot_and_save_cm(best['confusion_mean'], 'MC Média - Melhor Desempenho Global', f'{img_dir}/cm_global_best.png', class_names=labels)
    plot_and_save_cm(worst['confusion_mean'], 'MC Média - Pior Desempenho Global', f'{img_dir}/cm_global_worst.png', class_names=labels)

    # por kernel
    for kernel_id, data in kernel_results.items():
        k_name = kernel_str(kernel_id)
        if data.get('max', {}).get('confusion_mean') is not None:
            plot_and_save_cm(data['max']['confusion_mean'], f'MC Média - Melhor {k_name}', f'{img_dir}/cm_kernel_{k_name}_best.png', class_names=labels)
        if data.get('min', {}).get('confusion_mean') is not None:
            plot_and_save_cm(data['min']['confusion_mean'], f'MC Média - Pior {k_name}', f'{img_dir}/cm_kernel_{k_name}_worst.png', class_names=labels)

    # template HTML (usamos confusion_mean em porcentagem)
    html_template = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Resultados SVM</title>
        <style> /* estilos idênticos ou adaptados ao original */ 
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #fdfdfd; }
            .container { max-width: 1000px; margin: 0 auto; background-color: #fff; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .metrics-table { width: 100%; border-collapse: collapse; margin: 25px 0; }
            .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: middle; }
            .metrics-table th { background-color: #0056b3; color: white; text-align: center; }
            .result-section { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 5px solid #0056b3; }
            h1, h2 { color: #333; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }
            .best-result { background-color: #e7f3e7; }
            .worst-result { background-color: #fdeeee; }
            .cm-image { max-width: 350px; display: block; margin: 10px auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatório de Avaliação de Parâmetros SVM</h1>
            <div class="result-section">
                <h2>Resultados Globais</h2>
                <table class="metrics-table">
                    <tr class="best-result"><td>Melhor Kernel</td><td>{{ kernel_names[global_results.max.kernel_id] }}</td></tr>
                    <tr class="best-result"><td>Acurácia Média de Treino</td><td>{{ "%.2f"|format(global_results.max.accuracy_train) }}% &plusmn; {{ "%.2f"|format(global_results.max.std_train) }}%</td></tr>
                    <tr class="best-result"><td>Acurácia Média de Teste</td><td>{{ "%.2f"|format(global_results.max.accuracy_test) }}% &plusmn; {{ "%.2f"|format(global_results.max.std_test) }}%</td></tr>
                    <tr class="best-result"><td>Train time (s)</td><td>{{ "%.2f"|format(global_results.max.train_time_mean) }} &plusmn; {{ "%.2f"|format(global_results.max.train_time_std) }}</td></tr>
                    <tr class="best-result"><td>Test time (s)</td><td>{{ "%.2f"|format(global_results.max.test_time_mean) }} &plusmn; {{ "%.2f"|format(global_results.max.test_time_std) }}</td></tr>
                    <tr class="best-result"><td>Configuração (C, γ)</td><td>({{ global_results.max.cost }}, {{ global_results.max.gamma }})</td></tr>
                    <tr class="best-result"><td colspan="2"><img class="cm-image" src="svm_report_images/cm_global_best.png" alt="Matriz de Confusão - Melhor Global"></td></tr>

                    <tr class="worst-result"><td>Pior Kernel</td><td>{{ kernel_names[global_results.min.kernel_id] }}</td></tr>
                    <tr class="worst-result"><td>Acurácia Média de Treino</td><td>{{ "%.2f"|format(global_results.min.accuracy_train) }}% &plusmn; {{ "%.2f"|format(global_results.min.std_train) }}%</td></tr>
                    <tr class="worst-result"><td>Acurácia Média de Teste</td><td>{{ "%.2f"|format(global_results.min.accuracy_test) }}% &plusmn; {{ "%.2f"|format(global_results.min.std_test) }}%</td></tr>
                    <tr class="worst-result"><td>Train time (s)</td><td>{{ "%.2f"|format(global_results.min.train_time_mean) }} &plusmn; {{ "%.2f"|format(global_results.min.train_time_std) }}</td></tr>
                    <tr class="worst-result"><td>Test time (s)</td><td>{{ "%.2f"|format(global_results.min.test_time_mean) }} &plusmn; {{ "%.2f"|format(global_results.min.test_time_std) }}</td></tr>
                    <tr class="worst-result"><td>Configuração (C, γ)</td><td>({{ global_results.min.cost }}, {{ global_results.min.gamma }})</td></tr>
                    <tr class="worst-result"><td colspan="2"><img class="cm-image" src="svm_report_images/cm_global_worst.png" alt="Matriz de Confusão - Pior Global"></td></tr>
                </table>
            </div>
            <div class="result-section">
                <h2>Resumo por Kernel</h2>
                {% for kernel_id, data in kernel_results.items() %}
                <h3>Kernel: <strong>{{ kernel_names[kernel_id] }}</strong></h3>
                <table class="metrics-table">
                    <tr><th>Tipo</th><th>Acurácia Train</th><th>Acurácia Test</th><th>Train time (s)</th><th>Test time (s)</th></tr>
                    <tr>
                        <td>Melhor</td>
                        <td>{{ "%.2f"|format(data.max.accuracy_train) }}% &plusmn; {{ "%.2f"|format(data.max.std_train) }}%</td>
                        <td>{{ "%.2f"|format(data.max.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.max.std_test) }}%</td>
                        <td>{{ "%.2f"|format(data.max.train_time_mean) }} &plusmn; {{ "%.2f"|format(data.max.train_time_std) }}</td>
                        <td>{{ "%.2f"|format(data.max.test_time_mean) }} &plusmn; {{ "%.2f"|format(data.max.test_time_std) }}</td>
                    </tr>
                    <tr>
                        <td>Pior</td>
                        <td>{{ "%.2f"|format(data.min.accuracy_train) }}% &plusmn; {{ "%.2f"|format(data.min.std_train) }}%</td>
                        <td>{{ "%.2f"|format(data.min.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.min.std_test) }}%</td>
                        <td>{{ "%.2f"|format(data.min.train_time_mean) }} &plusmn; {{ "%.2f"|format(data.min.train_time_std) }}</td>
                        <td>{{ "%.2f"|format(data.min.test_time_mean) }} &plusmn; {{ "%.2f"|format(data.min.test_time_std) }}</td>
                    </tr>
                </table>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    """

    # preencher nomes de kernel
    kernel_names = {k: kernel_str(k) for k in kernel_results.keys()}
    # adicionar kernel_name ao global_results para fácil acesso no template
    global_results['max']['kernel_name'] = kernel_str(global_results['max']['kernel_id'])
    global_results['min']['kernel_name'] = kernel_str(global_results['min']['kernel_id'])

    template = Template(html_template)
    html_content = template.render(global_results=global_results, kernel_results=kernel_results, kernel_names=kernel_names)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\nRelatório HTML gerado com sucesso: '{output_file}'")
    except IOError as e:
        print(f"\nErro ao salvar o relatório HTML: {e}")

# ---------------------------
# Argument parsing e execução
# ---------------------------
def setOpts(argv):
    parser = argparse.ArgumentParser(description='Testador de Parâmetros SVM com Geração de Relatório HTML')
    parser.add_argument('-dataset', dest='dataset', action='store', default='heart_scale',
                        help='Nome do arquivo do conjunto de dados (formato LIBSVM)')
    parser.add_argument('-threshold', dest='threshold', action='store', type=float, default=0.1,
                        help='Limiar de correlação para seleção de características')
    args = parser.parse_args(argv)
    return args.dataset, args.threshold

if __name__ == "__main__":
    dataset, threshold = setOpts(sys.argv[1:])
    experiment = svmParameters()
    global_results, kernel_results = experiment.main(dataset, threshold)
    generate_html_report(global_results, kernel_results)
