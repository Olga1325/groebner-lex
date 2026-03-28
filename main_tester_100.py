import json
import os
import random
import time
import psutil
import pandas as pd
from ginv.monom import Monom
from ginv.poly import Poly
from ginv.gb import GB
from ginv.ginv import *
import itertools
from collections import Counter

VERY_QUICK = ['quadfor2', 'sparse5', 'hunecke', 'solotarev', 'chandra4', 'quadgrid', 'lorentz', 'liu', 'hemmecke', 'boon', 'chandra5', 'caprasse', 'issac97', 'hcyclic5', 'redcyc5', 'cyclic5', 'extcyc4', 'chemequ', 'uteshev_bikker', 'chandra6', 'geneig']
QUICK = ['chemequs', 'vermeer', 'camera1s', 'reimer4', 'redeco7', 'tangents', 'cassou', 'butcher', 'eco7', 'cohn2', 'dessin1', 'des18_3', 'hcyclic6', 'noon5', 'katsura6', 'cyclic6', 'butcher8', 'redcyc6', 'cpdm5', 'extcyc5']
MEDIUM = ['noon6', 'reimer5', 'kotsireas', 'assur44']
TOO_LONG = ['reimer8', 'reimer7', 'redeco12', 'redcyc8', 'redcyc7', 'noon9', 'noon8', 'mckay', 'mckay.gls50mod', 'katsura10', 'ilias13', 'ilias12', 'ilias_k_2', 'ilias_k_3', 'hf855', 'hcyclic8', 'hcyclic7', 'hawes4', 'hairer4', 'f966', 'f855']

def init(variables, order = Monom.TOPdeglex):
    Monom.init(variables)
    Monom.variables = variables.copy()
    Monom.zero = Monom(0 for _ in Monom.variables)
    Monom.cmp = order
    Poly.cmp = order
    for i in range(len(Monom.variables)):
        p = Poly()
        p.append([Monom(0 if l != i else 1 for l in range(len(Monom.variables))), 1])
        globals()[Monom.variables[i]] = p

def receiving_json(test_name, json_data, var_order=None, out=False):
    try:
        size = json_data["dimension"]
        if out:
            print(f"Тест для {test_name}, порядок переменных: {var_order}")

        variables = json_data["variables"]
        if var_order is None:
            var_order = variables.copy()

        init(var_order)

        equations = json_data['equations']
        eqs = []
        for eq in equations:
            eqs.append(eval(eq.replace('^', '**')))

        G = GB()
        G.algorithm2(eqs)

        leads = ', '.join(str(g.lm()) for g in G)

        data = {
            'time': str(G.time),
            'dimension': size,
            'crit1': str(G.crit1),
            'crit2': str(G.crit2),
            'leads': leads,
            'basis': str(G),
            'variable_order': var_order
        }

        if out:
            print(G)
            print("crit1 =", G.crit1, "crit2 =", G.crit2)
            print("time %.2f" % G.time)

        return data
    except Exception as e:
        print(f"Error processing JSON {test_name}: {e}")
        return None


def generate_var_orders(variables, n=5):
    # Генерирует n разных порядков переменных
    # Включает: оригинал, обратный, случайные, "через одну"

    orders = []
    n_vars = len(variables)

    # 1. Оригинальный порядок
    orders.append(variables.copy())

    # 2. Обратный порядок
    orders.append(variables[::-1])

    # 3. Через одну (если возможно)
    if n_vars > 2:
        odd = variables[::2]  # x1, x3, x5...
        even = variables[1::2]  # x2, x4, x6...
        mixed = odd + even
        if mixed != variables and mixed != variables[::-1]:
            orders.append(mixed)

    # 4. Случайные перестановки (до n-3 штук)
    while len(orders) < n:
        shuffled = variables.copy()
        random.shuffle(shuffled)
        if shuffled not in orders:
            orders.append(shuffled)

    # Убираем дубликаты и ограничиваем до n (если вдруг случайно повторился порядок)
    unique_orders = []
    for order in orders:
        if order not in unique_orders:
            unique_orders.append(order)
        if len(unique_orders) >= n:
            break

    return unique_orders[:n]


def generate_lex_orders(variables, total_runs=30, mode='balanced_first'):
    n = len(variables)
    orders = []

    if mode == 'balanced_first':
        total_orders = n * total_runs
        print(f"Режим balanced_first: каждая переменная будет на первой позиции ровно {total_runs} раз")
        print(f"Итого будет сгенерировано {total_orders} порядков\n")

        for first_var in variables:
            remaining = [v for v in variables if v != first_var]

            for _ in range(total_runs):
                shuffled_rem = remaining[:]
                random.shuffle(shuffled_rem)
                order = [first_var] + shuffled_rem
                orders.append(order)

        print(f"Сгенерировано {len(orders)} порядков (по {total_runs} на каждую ведущую переменную)")

    elif mode == 'balanced_all':
        print(f" Генерируем по {total_runs} порядков для каждой из {n} позиций: всего {total_runs * n} порядков\n")

        for pos in range(n):
            group = []
            print(f" Позиция {pos + 1}/{n}: {total_runs} порядков с балансом на этой позиции", end=" ")

            per = total_runs // n
            rem = total_runs % n

            for i, target_var in enumerate(variables):
                num = per + (1 if i < rem else 0)
                remaining = [v for v in variables if v != target_var]

                for _ in range(num):
                    order = [None] * n
                    order[pos] = target_var

                    shuffled = remaining[:]
                    random.shuffle(shuffled)
                    idx = 0
                    for p in range(n):
                        if p != pos:
                            order[p] = shuffled[idx]
                            idx += 1

                    group.append(order)

            counts = pd.DataFrame(0, index=variables, columns=[f'pos_{p + 1}' for p in range(n)])
            for order in group:
                for p, var in enumerate(order):
                    counts.loc[var, f'pos_{p + 1}'] += 1

            counts.to_csv(f'results_cyclic7_lex/position_counts_pos{pos + 1}.csv', sep=';')
            print("готово")
            orders.extend(group)

        print(f"\n Сгенерировано {len(orders)} порядков (по {total_runs} на каждую позицию)\n")

    elif mode == 'random':
        while len(orders) < total_runs:
            shuffled = variables[:]
            random.shuffle(shuffled)
            if shuffled not in orders:
                orders.append(shuffled)

    elif mode == 'old_style':
        base_orders = generate_var_orders(variables, n=5)
        while len(orders) < total_runs:
            orders.extend(base_orders)
        orders = orders[:total_runs]
        random.shuffle(orders)

    random.shuffle(orders)
    return orders

    if mode in ('random', 'old_style'):
        return orders[:total_runs]
    else:
        return orders


def count_and_report_positions(orders, variables, out_dir='results_geneig_lex'):

   # Подсчитываем, сколько раз каждая переменная оказалась на каждой позиции в списке порядков

    n = len(variables)
    # Создаём таблицу: строки — переменные, столбцы — позиции
    counts = pd.DataFrame(0, index=sorted(variables), columns=[f'pos_{i + 1}' for i in range(n)])

    for order in orders:
        for pos, var in enumerate(order):
            counts.loc[var, f'pos_{pos + 1}'] += 1

    print("Распределение переменных по позициям в сгенерированных порядках")
    print(counts)
    print(f"\nВсего порядков: {len(orders)}")

    # Сохраняем в csv
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f'{out_dir}/position_counts_{len(orders)}.csv'
    counts.to_csv(csv_path, sep=';')
    print(f"Таблица сохранена {csv_path}\n")

    return counts

def run_geneig_lex_experiments(total_runs=100, mode='balanced_first', seed=42, out_dir='results_geneig_lex'):

    random.seed(seed)

    test_name = 'geneig'
    out_dir = test_name + '_lex'
    json_path = f'json/{test_name}.json'

    if not os.path.exists(json_path):
        print(f"Файл {json_path} не найден!")
        return None

    json_data = json.load(open(json_path))
    variables = json_data["variables"]
    print(f"{test_name}: {len(variables)} переменных -> {variables}")

    os.makedirs(out_dir, exist_ok=True)
    orders = generate_lex_orders(variables, total_runs=total_runs, mode=mode)

    print(f"Сгенерировано {len(orders)} порядков (mode={mode})")

    count_and_report_positions(orders, variables, out_dir)

    results = []
    for idx, var_order in enumerate(orders, 1):
        first_var = var_order[0]
        print(f"[{idx:3d}/{len(orders)}] первая: {first_var:>3} порядок: {' '.join(var_order)}")

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024

        start_time = time.perf_counter()
        result = receiving_json(test_name, json_data, var_order=var_order, out=False)
        elapsed = time.perf_counter() - start_time

        mem_after = process.memory_info().rss / 1024 / 1024
        max_mem = max(mem_before, mem_after)

        if result is not None:
            crit_total = int(result['crit1']) + int(result['crit2'])
            entry = {
                'run_id': idx,
                'first_var': first_var,
                'order_str': '_'.join(var_order),
                'time_s': elapsed,
                'crit_total': crit_total,
                'max_memory_mb': max_mem,
                'leads': result.get('leads', ''),
                'variable_order': var_order
            }
            results.append(entry)

            with open(f'{out_dir}/{test_name}_run{idx:03d}.json', 'w', encoding='utf-8') as f:
                json.dump({**result, **entry}, f, ensure_ascii=False, indent=2)

    if results:
        df = pd.DataFrame(results)
        csv_name = f'{out_dir}/{test_name}_lex_{len(orders)}runs.csv'
        df.to_csv(csv_name, sep=';', index=False, encoding='utf-8')
        print(f"\nРезультаты сохранены -> {csv_name}")
        return df

    return None

if __name__ == '__main__':
    print("Запуск 100 лексикографических порядков")
    df = run_geneig_lex_experiments(
        total_runs=100,
        mode='balanced_first'
    )
    if df is not None:
        print("\nКраткая статистика:")
        print(df.groupby('first_var')['time_s'].agg(['count','mean','min','max','median']).round(2))