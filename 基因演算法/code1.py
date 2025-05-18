import numpy as np
import matplotlib.pyplot as plt
import random

# 目標函數
def target_function(x):
    return -15 * (np.sin(2*x))**2 - (x-2)**2 + 160

# 基因解碼（二進位轉十進位再映射到 x）
def decode(individual, bit_length):
    decimal = int("".join(str(bit) for bit in individual), 2)
    return -10 + (20 / (2**bit_length - 1)) * decimal

# 初始化族群
def initialize_population(pop_size, bit_length):
    return [np.random.randint(0, 2, bit_length).tolist() for _ in range(pop_size)]

# 輪盤法選擇 (回傳 index)
def roulette_wheel_selection_indices(fitness, num_selected):
    total_fitness = sum(fitness)
    pick = np.random.rand(num_selected) * total_fitness
    selected_indices = []
    for p in pick:
        current = 0
        for idx, f in enumerate(fitness):
            current += f
            if current >= p:
                selected_indices.append(idx)
                break
    return selected_indices

# 單點交配
def single_point_crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# 單點突變
def single_point_mutation(individual):
    point = np.random.randint(len(individual))
    individual[point] = 1 - individual[point]

# 遺傳演算法主程式
def genetic_algorithm(fitness_mode='direct',
                      pop_size=10, bit_length=10,
                      mutation_rate=0.01, crossover_rate=0.8,
                      max_gen=1000, init_pop=None,
                      f_opt=None):
    if init_pop is None:
        pop = initialize_population(pop_size, bit_length)
    else:
        pop = [ind.copy() for ind in init_pop]

    history_best = []
    history_avg = []
    avg_above_152_count = 0
    generation = 0

    while generation < max_gen:
        x_vals = np.array([decode(ind, bit_length) for ind in pop])
        fx_values = target_function(x_vals)
        best_fx = fx_values.max()
        best_idx = np.argmax(fx_values)
        best_x = x_vals[best_idx]

        history_best.append(best_fx)
        history_avg.append(fx_values.mean())

        # 收斂條件：average 大於 152 一次且 best 等於目標最佳解
        if history_avg[-1] > 152:
            avg_above_152_count += 1
        if avg_above_152_count >= 1 and abs(best_fx - f_opt) <= 0.01:
            print(f"Converged at generation {generation} (mode={fitness_mode}), best_fx={best_fx:.5f}, best_x={best_x:.5f}")
            break

        if generation >= max_gen:
            print(f"Reached maximum generations without convergence (mode={fitness_mode})")
            break

        if fitness_mode == 'direct':
            fitness = fx_values
        elif fitness_mode == 'square':
            fitness = fx_values**8
        elif fitness_mode == 'linear':
            fitness = 3 * fx_values + 50
        else:
            raise ValueError("Invalid fitness_mode")

        if fitness.min() < 0:
            fitness = fitness - fitness.min() + 1e-6

        num_sel = int(crossover_rate * pop_size)
        selected_indices = roulette_wheel_selection_indices(fitness, num_sel)
        selected = [pop[i] for i in selected_indices]
        survivors = [pop[i] for i in range(pop_size) if i not in selected_indices]

        random.shuffle(selected)
        children = []
        for i in range(0, len(selected), 2):
            if i+1 < len(selected):
                c1, c2 = single_point_crossover(selected[i], selected[i+1])
                children.extend([c1, c2])
        if children:
            single_point_mutation(children[np.random.randint(len(children))])

        pop = children + survivors
        generation += 1

    plt.figure()
    plt.plot(history_best, label='Best')
    plt.plot(history_avg, label='Average')
    plt.xlabel('Generation')
    plt.ylabel('f(x)')
    plt.title(f'Fitness Evolution (mode={fitness_mode})')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    POP_SIZE = 10
    BIT_LENGTH = 10
    MUTATION_RATE = 0.01
    CROSSOVER_RATE = 0.8
    MAX_GEN = 1000

    x_grid = np.arange(2**BIT_LENGTH)
    x_vals = -10 + (20 / (2**BIT_LENGTH - 1)) * x_grid
    y_vals = target_function(x_vals)
    f_opt = y_vals.max()

    plt.figure()
    plt.plot(x_vals, y_vals, label='Target function (decoded from binary)')
    idx_max = np.argmax(y_vals)
    plt.scatter(x_vals[idx_max], y_vals[idx_max], color='red', marker='o',
                s=50, label=f'Maximum ({x_vals[idx_max]:.3f}, {y_vals[idx_max]:.3f})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Objective function (1024 decoded points)')
    plt.legend()
    plt.grid()
    plt.show()

    init_pop = initialize_population(POP_SIZE, BIT_LENGTH)
    for mode in ['direct', 'square', 'linear']:
        genetic_algorithm(
            fitness_mode=mode,
            pop_size=POP_SIZE,
            bit_length=BIT_LENGTH,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
            max_gen=MAX_GEN,
            init_pop=init_pop,
            f_opt=f_opt
        )