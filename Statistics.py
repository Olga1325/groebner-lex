import json
import pandas as pd
import numpy as np
import os

OUT_DIR = 'geneig_lex'  # например

# Собираем все JSON-файлы из папки
files = [f for f in os.listdir(OUT_DIR) if f.endswith('.json') and 'geneig_run' in f]
print(f"Найдено {len(files)} JSON-файлов для анализа...")

# Загружаем данные из всех JSON
results = []
for filename in sorted(files):
    path = os.path.join(OUT_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Извлекаем нужные поля
        entry = {
            'run_id': data.get('run_id', int(filename.split('run')[1].split('.')[0])),
            'first_var': data.get('first_var'),
            'time_s': float(data.get('time_s', 0)),
            'crit1': int(data.get('crit1', 0)),
            'crit2': int(data.get('crit2', 0)),
            'crit_total': int(data.get('crit1', 0)) + int(data.get('crit2', 0)),
            'max_memory_mb': data.get('max_memory_mb', 0),
            'order_str': data.get('order_str'),
            'variable_order': data.get('variable_order'),
        }
        results.append(entry)

df = pd.DataFrame(results)
print(f"Загружено {len(df)} запусков.")

# Сохраняем всё в CSV
csv_filename = f'{OUT_DIR}/all_results_analysis.csv'
df.to_csv(csv_filename, sep=';', index=False, encoding='utf-8')
print(f"\nРезультаты сохранены в: {csv_filename}")

# Выводим нужную статистику для каждой переменной
all_vars = sorted(set(var for order_str in df['order_str'] for var in order_str.split('_')))
print("\nСтатистика по переменным")
for var in all_vars:
    # Собираем все запуски, где var встречается
    mask = df['order_str'].apply(lambda s: var in s.split('_'))
    sub_df = df[mask].copy()

    # Собираем время по позициям
    pos_times = [[] for _ in range(len(all_vars))]
    pos_counts = [0] * len(all_vars)

    for _, row in sub_df.iterrows():
        order = row['order_str'].split('_')
        for p, v in enumerate(order):
            if v == var:
                pos_times[p].append(row['time_s'])
                pos_counts[p] += 1

    print(f"\n{var}:")
    for p, count in enumerate(pos_counts, 1):
        if count > 0:
            times = pos_times[p - 1]
            print(
                f"pos {p}: {count} раз, медиана {np.median(times):.2f} с (min {min(times):.2f} – max {max(times):.2f})"
            )
        else:
            print(f"pos {p}: 0 раз")

# Топ-10 быстрых
print("\nТоп-10 самых быстрых")
print(df.nsmallest(10, 'time_s')[['first_var', 'time_s', 'crit_total', 'order_str']])

# Топ-10 медленных
print("\nТоп-10 самых медленных")
print(df.nlargest(10, 'time_s')[['first_var', 'time_s', 'crit_total', 'order_str']])

# Статистика по времени
print("\nСтатистика по времени")
print(df['time_s'].describe().round(2))

# Статистика по критическим парам
print("\nСтатистика по критическим парам (crit_total)")
print(df['crit_total'].describe().round(2))

# Статистика по памяти
print("\nСтатистика по памяти")
print(df['max_memory_mb'].describe().round(2))

print("\nАнализ завершён.")