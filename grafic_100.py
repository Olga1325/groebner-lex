import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

OUT_DIR = 'geneig_lex'
CSV_FILE = f'{OUT_DIR}/geneig_lex_600runs.csv'

df = pd.read_csv(CSV_FILE, sep=';')
df['run_id'] = df.index + 1

# Определяем переменные
first_order = df['order_str'].iloc[0].split('_')
all_vars = sorted(set(first_order))
n_vars = len(all_vars)
print(f"Переменные ({n_vars}): {all_vars}")

# Определяем размер группы
total_runs = len(df)
if total_runs % n_vars != 0:
    raise ValueError("Общее число запусков должно делиться на число переменных (группы равны)")
group_size = total_runs // n_vars
print(f"Группы: {n_vars} шт., по {group_size} запусков каждая")

# Внутри каждой группы: распределение переменных
per = group_size // n_vars
rem = group_size % n_vars
print(f"Внутри группы: per={per}, rem={rem} - первые {rem} переменных по {per+1}, остальные по {per}")

# Функция: по run_id - (target_var, target_pos)
def get_target_info(run_id):
    group_idx = (run_id - 1) // group_size  # 0-based группа - target_pos = group_idx + 1
    in_group = (run_id - 1) % group_size
    cum = [0]
    for i in range(n_vars):
        cnt = per + (1 if i < rem else 0)
        cum.append(cum[-1] + cnt)
    for i in range(n_vars):   # Находим, какой переменной соответствует in_group
        if in_group < cum[i + 1]:
            return all_vars[i], group_idx + 1  # target_var, target_pos (1-based)
    return all_vars[-1], group_idx + 1

# Применяем
target_info = df['run_id'].apply(get_target_info)
df['target_var'] = [t[0] for t in target_info]
df['target_pos'] = [t[1] for t in target_info]

# Проверка баланса
print("\nПроверка баланса в группе 1 (позиция 1):")
print(df[df['target_pos'] == 1]['target_var'].value_counts().sort_index())

# Строим графики по позициям

Y_MAX = 40

for pos in range(1, n_vars + 1):
    plt.figure(figsize=(11, 6))

    sub_df = df[df['target_pos'] == pos]

    # Подготовка данных
    var_order = []
    labels = []
    for var in all_vars:
        times = sub_df[sub_df['target_var'] == var]['time_s'].dropna().tolist()
        count = len(times)
        var_order.append(var)
        labels.append(f"{var}\n({count})")
    ax = sns.boxplot(
        x='target_var',
        y='time_s',
        hue='target_var',
        data=sub_df,
        order=var_order,
        palette="Set3",  # цветные
        legend=False
    )

    # Подписи
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    # Оформление
    plt.title(f'Время вычисления когда переменная стояла на позиции {pos}\n', fontsize=14, pad=15)
    plt.xlabel(f'Переменная на позиции {pos}', fontsize=12)
    plt.ylabel('Время, секунды', fontsize=12)

    plt.ylim(0, Y_MAX)
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=0)

    # Сохранение
    save_path = f'{OUT_DIR}/time_by_target_var_on_position_{pos}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.show()

    print(f"График для позиции {pos} сохранён: {save_path}")


# Определяем переменные из order_str
first_order = df['order_str'].iloc[0].split('_')
all_vars = sorted(set(first_order))
print("Переменные:", all_vars)

# Удалим явные аномалии — время > 100 с
df_clean = df[df['time_s'] <= 30]

# Строим 2 подграфика в одном окне
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. Полный диапазон (с выбросами)
sns.scatterplot(x='max_memory_mb', y='time_s', data=df, ax=ax1, alpha=0.6, s=50, color='steelblue')
sns.regplot(x='max_memory_mb', y='time_s', data=df, ax=ax1, scatter=False, color='red', ci=95, line_kws={'lw':2})
ax1.set_title('С выбросами', fontsize=12)
ax1.set_xlabel('Макс. память, МБ')
ax1.set_ylabel('Время, секунды')
ax1.grid(True, alpha=0.3)

# 2. Без выбросов
sns.scatterplot(x='max_memory_mb', y='time_s', data=df_clean, ax=ax2, alpha=0.6, s=50, color='darkgreen')
sns.regplot(x='max_memory_mb', y='time_s', data=df_clean, ax=ax2, scatter=False, color='red', ci=95, line_kws={'lw':2})
ax2.set_title('Без выбросов (time ≤ 100 с)', fontsize=12)
ax2.set_xlabel('Макс. память, МБ')
ax2.set_ylabel('Время, секунды')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 30)  # фиксированная шкала для детализации(можно поменять время)

plt.suptitle('Время вычисления vs Макс. память', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/time_vs_memory_clean_and_full.png', dpi=150)
plt.show()

# Выводим топ-10 и статистику по всем данным
print("\nТоп-10 самых быстрых:")
print(df.sort_values('time_s').head(10)[['first_var', 'time_s', 'crit_total', 'order_str']])

print("\nТоп-10 самых медленных порядков:")
print(df.sort_values('time_s', ascending=False).head(10)[['first_var', 'time_s', 'crit_total', 'order_str']])

print("\nСтатистика по времени:")
print(df['time_s'].describe().round(2))

print("\nСтатистика по памяти:")
print(df['max_memory_mb'].describe().round(2))

def plot_time_by_variable_per_position(df, out_dir=OUT_DIR, y_max=40):

    # Графики: для каждой позиции — время в зависимости от переменной
    if 'variable_order' not in df.columns:
        print("Ошибка: нет столбца 'variable_order'")
        return

    # Преобразуем строку в список, если нужно
    if isinstance(df['variable_order'].iloc[0], str):
        df['variable_order'] = df['variable_order'].apply(eval)

    n_vars = len(df['variable_order'].iloc[0])

    # Создаём столбцы pos_1 и тд
    for i in range(n_vars):
        df[f'pos_{i + 1}'] = df['variable_order'].apply(lambda x: x[i])

    print(f"\nПостроение {n_vars} графиков (Y max = {y_max} сек)...\n")

    for pos in range(1, n_vars + 1):
        pos_col = f'pos_{pos}'

        plt.figure(figsize=(11, 6))

        # Подготовка данных + количество появлений
        var_order = []
        labels = []
        for var in sorted(df[pos_col].unique()):
            count = (df[pos_col] == var).sum()
            var_order.append(var)
            labels.append(f"{var}\n({count})")

        ax = sns.boxplot(
            x=pos_col,
            y='time_s',
            hue=pos_col,
            data=df,
            order=var_order,
            palette="Set3",
            legend=False
        )

        # Установка подписей
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)

        # Оформление
        plt.title(f'Время вычисления когда переменная стояла на позиции {pos}\n'f'(все прогоны вместе)', fontsize=14, pad=15)
        plt.xlabel(f'Переменная на позиции {pos}', fontsize=12)
        plt.ylabel('Время, секунды', fontsize=12)

        plt.ylim(0, y_max)
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=0)

        # Сохранение
        save_path = f'{out_dir}/time_by_variable_at_position_{pos}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
        plt.show()

        print(f" Позиция {pos} сохранена: {save_path}")

    print(f"\n Создано {n_vars} графиков.")

plot_time_by_variable_per_position(df, out_dir=OUT_DIR, y_max=40)