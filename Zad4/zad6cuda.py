from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from RandomNumberGenerator import RandomNumberGenerator
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

def gen(seed, n, m):
    S = np.zeros((n, m))
    rand = RandomNumberGenerator(seed)
    for i in range(n):
        for j in range(m):
            S[i, j] = rand.nextInt(1, 29)
    return S.T

# Kernel CUDA do obliczania Cmax
cuda_calculate_cmax_kernel = """
__global__ void calculate_cmax_kernel(float *tasks, float *cmax_out, int m, int n, int num_perms, int *permutations) {
    int perm_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (perm_idx >= num_perms) return;

    // Każdy wątek oblicza Cmax dla jednej permutacji
    int *perm = permutations + perm_idx * n;

    // Alokuj tablicę C dla tej permutacji (w shared memory jeśli możliwe, inaczej lokalnie)
    extern __shared__ float shared_mem[];
    float *C = &shared_mem[threadIdx.x * m * n];

    // Oblicz Cmax dla tej permutacji
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int task_col = perm[j];
            float task_time = tasks[i * n + task_col];

            if (i == 0 && j == 0) {
                C[i * n + j] = task_time;
            } else if (i == 0) {
                C[i * n + j] = C[i * n + (j - 1)] + task_time;
            } else if (j == 0) {
                C[i * n + j] = C[(i - 1) * n + j] + task_time;
            } else {
                float prev_machine = C[(i - 1) * n + j];
                float prev_task = C[i * n + (j - 1)];
                C[i * n + j] = fmaxf(prev_machine, prev_task) + task_time;
            }
        }
    }

    // Ostatni element to Cmax
    cmax_out[perm_idx] = C[(m - 1) * n + (n - 1)];
}

__global__ void calculate_single_cmax_kernel(float *tasks, float *cmax_out, int m, int n) {
    // Kernel do obliczania pojedynczego Cmax
    extern __shared__ float C[];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float task_time = tasks[i * n + j];

            if (i == 0 && j == 0) {
                C[i * n + j] = task_time;
            } else if (i == 0) {
                C[i * n + j] = C[i * n + (j - 1)] + task_time;
            } else if (j == 0) {
                C[i * n + j] = C[(i - 1) * n + j] + task_time;
            } else {
                float prev_machine = C[(i - 1) * n + j];
                float prev_task = C[i * n + (j - 1)];
                C[i * n + j] = fmaxf(prev_machine, prev_task) + task_time;
            }
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cmax_out[0] = C[(m - 1) * n + (n - 1)];
    }
}
"""

mod = SourceModule(cuda_calculate_cmax_kernel)
calculate_cmax_kernel = mod.get_function("calculate_cmax_kernel")
calculate_single_cmax_kernel = mod.get_function("calculate_single_cmax_kernel")

def calculate_Cmax_cuda(Tasks):
    """Oblicza Cmax dla jednej permutacji zadań używając CUDA"""
    m, n = Tasks.shape
    tasks_gpu = gpuarray.to_gpu(Tasks.astype(np.float32).flatten())
    cmax_gpu = gpuarray.zeros(1, dtype=np.float32)

    shared_mem_size = m * n * 4  # 4 bytes per float

    calculate_single_cmax_kernel(
        tasks_gpu, cmax_gpu,
        np.int32(m), np.int32(n),
        block=(1, 1, 1),
        grid=(1, 1),
        shared=shared_mem_size
    )

    return float(cmax_gpu.get()[0])

def calculate_Cmax(Tasks):
    """Wersja CPU do porównania - pozostawiona z oryginalnego kodu"""
    m, n = Tasks.shape
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                C[i, j] = Tasks[i, j]
            elif i == 0:
                C[i, j] = C[i, j - 1] + Tasks[i, j]
            elif j == 0:
                C[i, j] = C[i - 1, j] + Tasks[i, j]
            else:
                C[i, j] = max(C[i - 1, j], C[i, j - 1]) + Tasks[i, j]
    return C[-1, -1]

def batch_calculate_cmax_cuda(Tasks, permutations):
    """
    Oblicza Cmax dla wielu permutacji równocześnie na GPU

    Args:
        Tasks: macierz zadań [m x n]
        permutations: lista permutacji, każda jako lista indeksów

    Returns:
        lista wartości Cmax dla każdej permutacji
    """
    m, n = Tasks.shape
    num_perms = len(permutations)

    if num_perms == 0:
        return []

    # Przygotuj dane
    tasks_flat = Tasks.astype(np.float32).flatten()
    perms_flat = np.array(permutations, dtype=np.int32).flatten()

    # Alokuj pamięć na GPU
    tasks_gpu = gpuarray.to_gpu(tasks_flat)
    perms_gpu = gpuarray.to_gpu(perms_flat)
    cmax_gpu = gpuarray.zeros(num_perms, dtype=np.float32)

    # Konfiguracja kernela
    threads_per_block = min(256, num_perms)
    blocks = (num_perms + threads_per_block - 1) // threads_per_block
    shared_mem_size = threads_per_block * m * n * 4  # 4 bytes per float

    # Uruchom kernel
    try:
        calculate_cmax_kernel(
            tasks_gpu, cmax_gpu,
            np.int32(m), np.int32(n), np.int32(num_perms),
            perms_gpu,
            block=(threads_per_block, 1, 1),
            grid=(blocks, 1),
            shared=shared_mem_size
        )

        # Pobierz wyniki
        results = cmax_gpu.get()
        return results.tolist()
    except cuda.Error as e:
        print(f"CUDA Error: {e}")
        # Fallback do CPU jeśli CUDA zawiedzie
        results = []
        for perm in permutations:
            cmax = calculate_Cmax(Tasks[:, perm])
            results.append(cmax)
        return results

# ===================================================================
# ================= Wyliczenie permutacji startowej =================
def alg_Johnsona2_for_2machine(tasks):
    l = 0
    m, n = tasks.shape
    k = n - 1
    N = [i for i in range(n)]
    pi = [None for _ in range(n)]
    NewTasks = deepcopy(tasks)
    while len(N) > 0:
        min_value = np.min(tasks)
        indices = np.where(tasks == min_value)
        j_star = indices[1][0]
        if tasks[0, j_star] < tasks[1, j_star]:
            pi[l] = int(j_star)
            l += 1
        else:
            pi[k] = int(j_star)
            k -= 1
        N.remove(j_star)
        tasks[:, j_star] = np.inf
    return pi

def permuteTasks(pi, tasks):
    tasks_toReturn = np.zeros_like(tasks)
    for M in range(tasks.shape[0]):
        for i in range(len(pi)):
            tasks_toReturn[M][i] = tasks[M][pi[i]]
    return tasks_toReturn

def alf_Johnson(tasks):
    m, n = tasks.shape
    if m == 2:
        return alg_Johnsona2_for_2machine(tasks)
    else:
        tasks_transformed = np.zeros((2, n))
        for i in range(n):
            tasks_transformed[0, i] = np.sum(tasks[:m - 1, i])
            tasks_transformed[1, i] = np.sum(tasks[1:m, i])
        pi = alg_Johnsona2_for_2machine(tasks_transformed)
        return pi

# ===================================================================
# ===================== Tabu Search z CUDA ==========================
def generate_block_neighbors(permutation, block_size=1):
    permutation = np.array(permutation)
    neighbors = []
    n = len(permutation)
    for i in range(n - block_size):
        for j in range(i + block_size, n - block_size + 1):
            new_perm = permutation.copy()
            temp = new_perm[i:i+block_size].copy()
            new_perm[i:i+block_size] = new_perm[j:j+block_size]
            new_perm[j:j+block_size] = temp
            neighbors.append((new_perm.tolist(), (i, j)))
    return neighbors

def tabuSearch_sequential(Tasks, start_permutation=None, stopValue=100, tabu_tenure=7, block_size=1):
    """
    Całkowicie sekwencyjna wersja Tabu Search (CPU).
    """
    n = Tasks.shape[1]
    if start_permutation is None:
        current_perm = list(range(n))
    else:
        current_perm = start_permutation.copy()
    current_tasks = Tasks[:, current_perm]
    current_cmax = calculate_Cmax(current_tasks)
    best_perm = current_perm.copy()
    best_cmax = current_cmax
    tabu_list = []
    cmax_history = [current_cmax]

    for iteration in range(stopValue):
        neighbors = generate_block_neighbors(current_perm, block_size)
        best_neighbor = None
        best_neighbor_cmax = float('inf')
        best_move = None

        for neighbor_perm, move in neighbors:
            if move not in tabu_list or calculate_Cmax(Tasks[:, neighbor_perm]) < best_cmax:
                cmax = calculate_Cmax(Tasks[:, neighbor_perm])
                if cmax < best_neighbor_cmax:
                    best_neighbor = neighbor_perm
                    best_neighbor_cmax = cmax
                    best_move = move

        if best_neighbor is None:
            break

        current_perm = best_neighbor
        current_cmax = best_neighbor_cmax
        cmax_history.append(current_cmax)

        if current_cmax < best_cmax:
            best_perm = current_perm.copy()
            best_cmax = current_cmax

        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return best_perm, best_cmax, cmax_history

def tabuSearch_cuda(Tasks, start_permutation=None, stopValue=100, tabu_tenure=7, block_size=1, batch_size=256):
    """
    Tabu Search z wykorzystaniem CUDA do równoległej ewaluacji sąsiadów na GPU.

    Args:
        Tasks: macierz zadań
        start_permutation: permutacja startowa
        stopValue: liczba iteracji
        tabu_tenure: długość listy tabu
        block_size: rozmiar bloku do zamiany
        batch_size: rozmiar batcha do przetwarzania na GPU (domyślnie 256)
    """
    n = Tasks.shape[1]
    if start_permutation is None:
        current_perm = list(range(n))
    else:
        current_perm = start_permutation.copy()

    current_tasks = Tasks[:, current_perm]
    current_cmax = calculate_Cmax_cuda(current_tasks)
    best_perm = current_perm.copy()
    best_cmax = current_cmax
    tabu_list = []
    cmax_history = [current_cmax]

    for iteration in range(stopValue):
        # Wygeneruj listę sąsiedztwa
        neighbors = generate_block_neighbors(current_perm, block_size)

        # Filtruj sąsiadów zgodnie z listą tabu (z warunkiem aspiracji)
        valid_neighbors = []
        for neighbor_perm, move in neighbors:
            if move not in tabu_list or calculate_Cmax(Tasks[:, neighbor_perm]) < best_cmax:
                valid_neighbors.append((neighbor_perm, move))

        if not valid_neighbors:
            break

        # Podziel na batche i oblicz Cmax na GPU
        all_cmax_values = []
        for i in range(0, len(valid_neighbors), batch_size):
            batch = valid_neighbors[i:i+batch_size]
            perms = [perm for perm, _ in batch]

            # Oblicz Cmax dla batcha permutacji na GPU
            batch_cmax = batch_calculate_cmax_cuda(Tasks, perms)
            all_cmax_values.extend(batch_cmax)

        # Znajdź najlepszego sąsiada
        best_idx = np.argmin(all_cmax_values)
        best_neighbor, best_move = valid_neighbors[best_idx]
        best_neighbor_cmax = all_cmax_values[best_idx]

        # Aktualizuj bieżące rozwiązanie
        current_perm = best_neighbor
        current_cmax = best_neighbor_cmax
        cmax_history.append(current_cmax)

        # Aktualizuj najlepsze globalne rozwiązanie
        if current_cmax < best_cmax:
            best_perm = current_perm.copy()
            best_cmax = current_cmax

        # Aktualizuj listę tabu
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return best_perm, best_cmax, cmax_history

# ===================================================================
if __name__ == "__main__":
    import time

    # Informacje o GPU
    print("="*60)
    print("INFORMACJE O GPU")
    print("="*60)
    try:
        print(f"Urządzenie CUDA: {pycuda.autoinit.device.name()}")
        print(f"Compute Capability: {pycuda.autoinit.device.compute_capability()}")
        print(f"Pamięć globalna: {pycuda.autoinit.device.total_memory() / 1024**3:.2f} GB")
    except Exception as e:
        print(f"Błąd podczas pobierania informacji o GPU: {e}")
    print("="*60)

    tasks = gen(123213, 20, 5)
    start_perm = alf_Johnson(tasks)
    print("\nPermutacja z algorytmu Johnsona:", start_perm)

    # Test sekwencyjny (CPU)
    print("\n=== Tabu Search SEKWENCYJNIE (CPU) ===")
    start_time = time.time()
    best_perm_seq, best_cmax_seq, cmax_history_seq = tabuSearch_sequential(
        Tasks=tasks,
        start_permutation=start_perm,
        stopValue=1500,
        tabu_tenure=7,
        block_size=3
    )
    sequential_time = time.time() - start_time
    print("Najlepsza permutacja:", best_perm_seq)
    print("Najlepsze Cmax:", best_cmax_seq)
    print(f"Czas wykonania (CPU): {sequential_time:.2f} sekund")

    # Test z CUDA
    print("\n=== Tabu Search z CUDA (GPU) ===")
    start_time = time.time()
    best_perm_cuda, best_cmax_cuda, cmax_history_cuda = tabuSearch_cuda(
        Tasks=tasks,
        start_permutation=start_perm,
        stopValue=1500,
        tabu_tenure=7,
        block_size=3,
        batch_size=512  # Zwiększony batch size dla Tesla T4
    )
    cuda_time = time.time() - start_time
    print("Najlepsza permutacja:", best_perm_cuda)
    print("Najlepsze Cmax:", best_cmax_cuda)
    print(f"Czas wykonania (CUDA): {cuda_time:.2f} sekund")

    # Porównanie wydajności
    print("\n" + "="*60)
    print("PODSUMOWANIE WYDAJNOŚCI")
    print("="*60)
    print(f"Czas CPU (sekwencyjny):              {sequential_time:.2f} s")
    print(f"Czas GPU (CUDA):                     {cuda_time:.2f} s")
    speedup = sequential_time / cuda_time
    print(f"Przyspieszenie GPU:                  {speedup:.2f}x")
    print(f"Jakość rozwiązania CPU:              {best_cmax_seq}")
    print(f"Jakość rozwiązania GPU:              {best_cmax_cuda}")
    print("="*60)

    # Wizualizacja (opcjonalna)
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(cmax_history_seq, marker='o', label='CPU')
    # plt.title("Spadek wartości Cmax - CPU")
    # plt.xlabel("Iteracja")
    # plt.ylabel("Cmax")
    # plt.grid(True)
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(cmax_history_cuda, marker='o', label='CUDA', color='green')
    # plt.title("Spadek wartości Cmax - CUDA")
    # plt.xlabel("Iteracja")
    # plt.ylabel("Cmax")
    # plt.grid(True)
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
