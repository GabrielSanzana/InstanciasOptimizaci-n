"""
Integrantes: Patricio Figueroa, Gabriel Sanzana, Kavon Kermani
"""

import random
from random import randrange, sample, shuffle
import math
import numpy as np
import torch
import datetime # Importar el módulo datetime
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# --- CONSTANTES GLOBALES (para mayor claridad y posible ajuste) ---
TOP_CANDIDATES_FOR_ITEM_OPS = 10 # Número de candidatos a considerar en las operaciones de añadir/quitar ítems en 'vecino'

# --- PARSEADOR DE ARCHIVOS TTP ---
def parsear_ttp(ruta_archivo):
    capacidad = min_vel = max_vel = renta = None
    coords = {}; items = []
    seccion = None
    with open(ruta_archivo, encoding='utf-8') as f:
        for linea in map(str.strip, f):
            if not linea or linea.startswith(('PROBLEM NAME', 'KNAPSACK DATA', 'DIMENSION', 'NUMBER OF ITEMS')): continue
            mayus = linea.upper()
            if 'CAPACITY' in mayus: capacidad = float(linea.split(':')[1])
            elif 'MIN SPEED' in mayus: min_vel = float(linea.split(':')[1])
            elif 'MAX SPEED' in mayus: max_vel = float(linea.split(':')[1])
            elif 'RENTING' in mayus: renta = float(linea.split(':')[1])
            elif 'NODE_COORD' in mayus: seccion = 'coords'
            elif 'ITEMS SECTION' in mayus: seccion = 'items'
            elif seccion == 'coords':
                partes = linea.split(); coords[int(partes[0])] = (float(partes[1]), float(partes[2]))
            elif seccion == 'items':
                partes = linea.split(); items.append((int(partes[0]), float(partes[1]), float(partes[2]), int(partes[3])))
    return items, coords, capacidad, min_vel, max_vel, renta


def construir_estructuras_cuda(items, coords_ciudades, device='cuda'):
    ciudades = sorted(coords_ciudades)
    id_a_indice = {cid: i for i, cid in enumerate(ciudades)}
    indice_a_id = {i: cid for i, cid in enumerate(ciudades)}
    num_ciudades, num_items = len(ciudades), len(items)

    # Para guardar información de items por ciudad (listas de tuplas, se mantienen en CPU)
    # Estas listas se usan para obtener los valores de p y w, que luego se convierten a tensores.
    items_por_ciudad_list = [[] for _ in range(num_ciudades)]

    # Pre-calcular el número máximo de ítems en cualquier ciudad para dimensionar los tensores
    # Esto es crucial para crear tensores 2D uniformes
    max_items_per_city_overall = 0 # El máximo global de ítems en cualquier ciudad
    temp_counts = [0] * num_ciudades
    for _, _, _, cid in items:
        idx_ciudad = id_a_indice[cid]
        temp_counts[idx_ciudad] += 1
    if temp_counts:
        max_items_per_city_overall = max(temp_counts)

    # Inicializar tensores para beneficios, pesos y máscaras con el tamaño máximo
    beneficios_tensor_global = torch.full((num_ciudades, max_items_per_city_overall), 0.0, device=device, dtype=torch.float32)
    pesos_tensor_global = torch.full((num_ciudades, max_items_per_city_overall), 0.0, device=device, dtype=torch.float32)
    # Esta máscara es para indicar qué posiciones en los tensores anteriores son ítems reales (no padding)
    mask_items_existentes = torch.full((num_ciudades, max_items_per_city_overall), False, dtype=torch.bool, device=device)

    contador_items_por_ciudad = [0]*num_ciudades

    for i_global, (id_item, ganancia, peso, cid) in enumerate(items):
        idx_ciudad = id_a_indice[cid]
        pos_local = contador_items_por_ciudad[idx_ciudad]
        beneficios_tensor_global[idx_ciudad, pos_local] = ganancia
        pesos_tensor_global[idx_ciudad, pos_local] = peso
        mask_items_existentes[idx_ciudad, pos_local] = True
        items_por_ciudad_list[idx_ciudad].append((i_global, ganancia, peso)) # Store tuple (global_idx, profit, weight)
        contador_items_por_ciudad[idx_ciudad] += 1

    # Tensor en GPU con la cantidad REAL de ítems en cada ciudad (sin padding)
    num_items_per_city_tensor = torch.tensor(contador_items_por_ciudad, dtype=torch.int32, device=device)

    # Convertir coords a tensor CUDA
    coords_arr = torch.tensor([coords_ciudades[cid] for cid in ciudades], dtype=torch.float32, device=device)

    # Matriz de distancias usando broadcasting en cuda
    matriz_distancias = torch.cdist(coords_arr.unsqueeze(0), coords_arr.unsqueeze(0)).squeeze(0)

    # Devolvemos los tensores globales de beneficios y pesos, y la máscara de ítems existentes
    # junto con las estructuras de CPU que todavía son necesarias
    return (ciudades, items_por_ciudad_list, id_a_indice, indice_a_id,
            num_ciudades, num_items, matriz_distancias, beneficios_tensor_global, pesos_tensor_global,
            mask_items_existentes, num_items_per_city_tensor, max_items_per_city_overall)


def insercion_mas_lejano_dp_cuda(matriz_dist):
    # matriz_dist es tensor float CUDA (nxn)
    n = matriz_dist.size(0)
    if n < 2:
        return [0]

    # Paso 1: Par de ciudades más lejano para iniciar tour
    max_val, max_idx = torch.max(matriz_dist.view(-1), 0)
    i, j = divmod(max_idx.item(), n)
    tour = [i, j, i]  # tour circular inicial

    no_visitadas = set(range(n)) - {i, j}

    def mejor_posicion_dp(ciudad):
        # Para cada arista (a,b) en tour, calcular costo de inserción
        a = torch.tensor(tour[:-1], device=matriz_dist.device)
        b = torch.tensor(tour[1:], device=matriz_dist.device)
        # Costo de inserción para cada posición
        costos = matriz_dist[a, ciudad] + matriz_dist[ciudad, b] - matriz_dist[a, b]
        # Encuentra la posición con mínimo costo
        min_val, min_idx = torch.min(costos, 0)
        return min_idx.item() + 1  # posición en tour para insertar

    while no_visitadas:
        # Para cada ciudad no visitada, calcular distancia mínima al tour
        tour_tensor = torch.tensor(tour[:-1], device=matriz_dist.device)
        no_visitadas_list = list(no_visitadas)

        c_tensor = torch.tensor(no_visitadas_list, device=matriz_dist.device)
        dists_to_tour_nodes = matriz_dist[c_tensor[:, None], tour_tensor[None, :]]
        min_dists = dists_to_tour_nodes.min(dim=1).values # Get min distance for each unvisited city

        # Find the city with the maximum minimum distance to the tour
        ciudad_a_aniadir_idx_local = torch.argmax(min_dists).item()
        ciudad = no_visitadas_list[ciudad_a_aniadir_idx_local]

        # Insertar ciudad en mejor posición
        pos = mejor_posicion_dp(ciudad)
        tour.insert(pos, ciudad)
        no_visitadas.remove(ciudad)

    return tour

def greedy_mochila_inicial_por_ciudad_cuda(items_por_ciudad_list, capacidad_maxima, max_items_ciudad,
                                           beneficios_tensor_global, pesos_tensor_global, mask_items_existentes, device='cuda'):

    num_ciudades = len(items_por_ciudad_list)
    # La máscara inicial debe ser un tensor CUDA
    mascara_seleccion = torch.full_like(mask_items_existentes, False, device=device)

    peso_actual_t = torch.tensor(0.0, device=device)
    conteo_por_ciudad_t = torch.zeros(num_ciudades, dtype=torch.int32, device=device)

    candidatos_cpu = [] # Usamos una lista de Python para la lógica de muestreo y ordenamiento

    # Construir lista de candidatos con ratio ajustado (en CPU para simplicidad)
    for indice_ciudad, ciudad_items_data in enumerate(items_por_ciudad_list):
        if len(ciudad_items_data) > 0:
            # Trabajar con tensores para calcular ratio_promedio
            beneficios_ciudad_t = beneficios_tensor_global[indice_ciudad, :len(ciudad_items_data)]
            pesos_ciudad_t = pesos_tensor_global[indice_ciudad, :len(ciudad_items_data)]
            ratio_promedio_t = beneficios_ciudad_t.sum() / pesos_ciudad_t.sum() if pesos_ciudad_t.sum() > 0 else torch.tensor(0.0, device=device)
            ratio_promedio = ratio_promedio_t.item() # Convert to scalar for CPU list

            for indice_local, (idx_global, profit, peso) in enumerate(ciudad_items_data):
                ratio = (profit ** 1.5) / peso if peso > 0 else float('inf')
                ratio_ajustado = ratio if ratio >= 0.8 * ratio_promedio else 0.0
                candidatos_cpu.append((ratio_ajustado, indice_ciudad, indice_local, profit, peso))

    shuffle(candidatos_cpu)
    candidatos_cpu.sort(key=lambda x: x[0], reverse=True)
    TOP = 10

    for i in range(len(candidatos_cpu)):
        # Seleccionar un subconjunto de los mejores candidatos para la aleatoriedad
        opciones_viables_cpu = []
        for j in range(TOP):
            if i + j < len(candidatos_cpu):
                ratio, cidx, ilidx, benef, wgt = candidatos_cpu[i+j]
                # Check conditions using tensor values
                if ratio > 0 and \
                   not mascara_seleccion[cidx, ilidx].item() and \
                   (peso_actual_t + wgt).item() <= capacidad_maxima and \
                   (conteo_por_ciudad_t[cidx] < max_items_ciudad).item(): # Ensure it's not already at max for city
                    opciones_viables_cpu.append((ratio, cidx, ilidx, benef, wgt))

        if not opciones_viables_cpu:
            break

        # Randomly select one from viable options
        ratio_sel, ciudad_sel, item_local_sel, beneficio_sel, peso_sel = random.choice(opciones_viables_cpu)

        # Update masks and weights on GPU
        mascara_seleccion[ciudad_sel, item_local_sel] = True
        peso_actual_t += peso_sel
        conteo_por_ciudad_t[ciudad_sel] += 1

    # Devuelve la máscara en la GPU
    return mascara_seleccion, peso_actual_t.item()


def info_items_seleccionados_cuda(mascara_seleccion, beneficios_tensor_global, pesos_tensor_global, mask_items_existentes):
    # mascara_seleccion es ahora un tensor 2D de booleanos en GPU
    # mask_items_existentes es para asegurar que solo consideramos ítems que realmente existen en el problema

    # Aplicar la máscara de ítems existentes
    seleccion_real = mascara_seleccion & mask_items_existentes

    # Sumar beneficios y pesos directamente en la GPU
    ganancia_t = (beneficios_tensor_global * seleccion_real).sum()
    peso_t = (pesos_tensor_global * seleccion_real).sum()

    beneficios_seleccionados = beneficios_tensor_global[seleccion_real]
    pesos_seleccionados = pesos_tensor_global[seleccion_real]

    ratio_prom = 0.0
    ratio_min = 0.0

    if pesos_seleccionados.numel() > 0: # Check if there are any selected items
        # Calculate ratios for selected items
        # Avoid division by zero: if weight is 0, ratio is inf
        ratios_tensor = torch.where(pesos_seleccionados > 0, beneficios_seleccionados / pesos_seleccionados, torch.tensor(float('inf'), device=pesos_seleccionados.device))

        # Calculate ratio_min directly on GPU
        ratio_min_t = torch.min(ratios_tensor)
        if torch.isinf(ratio_min_t):
            ratio_min = 0.0 # Handle case where all ratios are inf (all selected items have 0 weight)
        else:
            ratio_min = ratio_min_t.item()

        # Calculate ratio_prom on GPU, filtering out inf values
        ratios_tensor_finite = ratios_tensor[torch.isfinite(ratios_tensor)]
        if ratios_tensor_finite.numel() > 0:
            ratio_prom = ratios_tensor_finite.mean().item()
        else:
            ratio_prom = 0.0 # All selected items have 0 weight

    return ganancia_t.item(), peso_t.item(), ratio_prom, ratio_min


def funcion_objetivo_cuda(sol, datos_estructura, capacidad_t, v_min_t, v_max_t, costo_alq_t, retorno_por_nodo=False, device='cuda'):
    tour, mascara_seleccion = sol # mascara_seleccion es un tensor 2D de CUDA
    (ciudades, items_por_ciudad_list, id_a_idx, idx_a_id,
     n_ciudades, n_items_dummy, dist_matrix, beneficios_tensor_global, pesos_tensor_global, mask_items_existentes,
     num_items_per_city_tensor_dummy, max_items_per_city_overall_dummy) = datos_estructura

    # Apply the mask of existing items to the selection mask
    seleccion_real = mascara_seleccion & mask_items_existentes

    # Beneficio y peso total del knapsack (vectorizado)
    beneficio_total_t = (beneficios_tensor_global * seleccion_real).sum()
    peso_total_t = (pesos_tensor_global * seleccion_real).sum()

    if peso_total_t.item() > capacidad_t.item(): # Comparar valores escalares
        # Penalizar si se excede la capacidad
        return (-float('inf'), beneficio_total_t.item(), peso_total_t.item(), 0.0, 0.0)

    tour_tensor = torch.tensor(tour, dtype=torch.long, device=device)

    # Vectorized calculation of weights picked up at each city in the tour
    total_peso_por_ciudad_seleccionado = (pesos_tensor_global * seleccion_real).sum(dim=1)

    # Gather weights for cities in the tour order
    pesos_en_ciudades_tour_order = total_peso_por_ciudad_seleccionado[tour_tensor]

    # Calculate cumulative sum of weights along the tour
    pesos_acumulados_en_ruta_t = torch.cumsum(pesos_en_ciudades_tour_order, dim=0)

    # Prepare indices for all segments, including closing the loop
    origins_indices = tour_tensor[:-1]
    destinations_indices = tour_tensor[1:]

    # Add the last segment from last city back to the first city
    origins_indices = torch.cat((origins_indices, tour_tensor[-1:]))
    destinations_indices = torch.cat((destinations_indices, tour_tensor[:1]))

    # Distances for all segments (vectorizado)
    dists_segments = dist_matrix[origins_indices, destinations_indices]

    # Weights for each segment (cumulative weight at the start of the segment)
    weights_segments = pesos_acumulados_en_ruta_t

    # Calculate velocities for all segments (vectorizado)
    vel_segments = torch.max(
        v_max_t - (weights_segments / capacidad_t) * (v_max_t - v_min_t),
        v_min_t
    )

    # Calculate time for all segments (vectorizado)
    time_segments = dists_segments / vel_segments

    tiempo_acumulado = time_segments.sum()
    distancia_acumulada = dists_segments.sum()

    valor_obj = beneficio_total_t.item() - costo_alq_t.item() * tiempo_acumulado.item()
    ratio_bp = (beneficio_total_t / peso_total_t).item() if peso_total_t.item() > 0 else 0
    vel_media = distancia_acumulada.item() / tiempo_acumulado.item() if tiempo_acumulado.item() > 0 else 0

    if retorno_por_nodo:
        # Esta rama no es crítica para el rendimiento en la búsqueda, se mantiene como está
        hist_benef = [0] * len(tour)
        hist_peso = [0] * len(tour)
        hist_vel = [0] * len(tour)
        hist_ratio = [0] * len(tour)
        hist_dist = [0] * len(tour)
        hist_obj = [0] * len(tour)
        return valor_obj, beneficio_total_t.item(), peso_total_t.item(), ratio_bp, vel_media, hist_benef, hist_peso, hist_vel, hist_ratio, hist_dist, hist_obj
    else:
        return valor_obj, beneficio_total_t.item(), peso_total_t.item(), ratio_bp, vel_media


def vecino(solucion, datos_estructura, capacidad_t, v_min_t, v_max_t, arriendo_t, temp_actual, max_items_dinamico):
    ruta_actual, mascara_actual_gpu = solucion
    (ciudades_ordenadas, items_por_ciudad_list_dummy, id_a_idx, idx_a_id, # dummy para items_por_ciudad_list
     n_ciudades, num_items_total_dummy, dist_matrix, beneficios_tensor_global, pesos_tensor_global, mask_items_existentes,
     num_items_per_city_tensor, max_items_per_city_overall) = datos_estructura

    nueva_ruta = ruta_actual[:]
    nueva_mascara_gpu = mascara_actual_gpu.clone().detach()

    ganancia_total_t, peso_total_item, _, ratio_min = \
        info_items_seleccionados_cuda(nueva_mascara_gpu, beneficios_tensor_global, pesos_tensor_global, mask_items_existentes)

    rand_val = torch.rand(1, device='cuda').item()

    if rand_val < 0.4 and n_ciudades > 3:
        # Route permutation
        indices = torch.sort(torch.randint(1, n_ciudades, (3,), device='cuda'))[0].tolist()
        i0, i1, i2 = indices
        nueva_ruta[i0], nueva_ruta[i1], nueva_ruta[i2] = nueva_ruta[i1], nueva_ruta[i2], nueva_ruta[i0]

    else:
        # Get count of selected items per city (on GPU)
        suma_por_ciudad_t = nueva_mascara_gpu.sum(dim=1)

        if rand_val < 0.7:  # Prefer adding items
            # Ciudades que no están al máximo dinámico Y tienen ítems NO SELECCIONADOS
            ciudades_candidatas_mask_gpu = (suma_por_ciudad_t < max_items_dinamico) & (num_items_per_city_tensor > suma_por_ciudad_t)
            ciudades_candidatas_indices_gpu = torch.nonzero(ciudades_candidatas_mask_gpu, as_tuple=True)[0]

            if ciudades_candidatas_indices_gpu.numel() > 0:
                ciudad_idx_sel = ciudades_candidatas_indices_gpu[torch.randint(0, ciudades_candidatas_indices_gpu.numel(), (1,), device='cuda')].item()

                mascara_ciudad_sel_t = nueva_mascara_gpu[ciudad_idx_sel]
                beneficios_ciudad_t = beneficios_tensor_global[ciudad_idx_sel]
                pesos_ciudad_t = pesos_tensor_global[ciudad_idx_sel]

                mask_existentes_en_ciudad = mask_items_existentes[ciudad_idx_sel]

                # Filtra ítems no seleccionados Y existentes en esta ciudad
                mask_no_seleccionados_y_existentes = ~mascara_ciudad_sel_t & mask_existentes_en_ciudad

                beneficios_ns = beneficios_ciudad_t[mask_no_seleccionados_y_existentes]
                pesos_ns = pesos_ciudad_t[mask_no_seleccionados_y_existentes]
                # Los índices locales corresponden a las columnas de la matriz global
                indices_locales_ns = torch.arange(max_items_per_city_overall, device='cuda', dtype=torch.long)[mask_no_seleccionados_y_existentes]

                if beneficios_ns.numel() > 0:
                    ratios_ns = torch.where(pesos_ns > 0, beneficios_ns / pesos_ns, torch.tensor(float('inf'), device='cuda'))

                    # Filtra por capacidad: solo los que no excederían el peso máximo de la mochila
                    valid_for_capacity_mask = (pesos_ns + peso_total_item <= capacidad_t.item()) # .item() porque peso_total_item es float

                    beneficios_ns_valid = beneficios_ns[valid_for_capacity_mask]
                    pesos_ns_valid = pesos_ns[valid_for_capacity_mask]
                    ratios_ns_valid = ratios_ns[valid_for_capacity_mask]
                    indices_locales_ns_valid = indices_locales_ns[valid_for_capacity_mask]

                    if indices_locales_ns_valid.numel() > 0:
                        # Ordena los candidatos válidos por ratio (descendente)
                        sorted_ratios, sorted_indices_local_valid = torch.sort(ratios_ns_valid, descending=True)

                        # Limita a un número pequeño de candidatos para la iteración secuencial
                        num_candidates_to_check = min(sorted_ratios.numel(), TOP_CANDIDATES_FOR_ITEM_OPS)

                        temp_item = temp_actual / 10.0 + 1e-5
                        temp_item = min(temp_item, 10.0)

                        for k_idx in range(num_candidates_to_check):
                            ratio_val = sorted_ratios[k_idx].item()
                            original_local_idx = indices_locales_ns_valid[sorted_indices_local_valid[k_idx]].item()
                            p_val = beneficios_tensor_global[ciudad_idx_sel, original_local_idx].item()
                            w_val = pesos_tensor_global[ciudad_idx_sel, original_local_idx].item()

                            if ratio_min == 0.0 or ratio_val >= ratio_min:
                                nueva_mascara_gpu[ciudad_idx_sel, original_local_idx] = True
                                peso_total_item += w_val
                                break # Se añadió un ítem, sale del bucle de candidatos
                            else:
                                delta = (ratio_val - ratio_min) / (ratio_min + 1e-9)
                                prob = math.exp(delta / temp_item)
                                if temp_item > 1e-10 and torch.rand(1, device='cuda').item() < prob:
                                    nueva_mascara_gpu[ciudad_idx_sel, original_local_idx] = True
                                    peso_total_item += w_val
                                    break # Se añadió un ítem, sale del bucle de candidatos
        else: # Removing items
            # Identificar todos los ítems seleccionados en GPU
            selected_global_indices_flat = torch.nonzero(nueva_mascara_gpu & mask_items_existentes, as_tuple=True)

            if selected_global_indices_flat[0].numel() > 0:
                selected_city_indices = selected_global_indices_flat[0]
                selected_item_local_indices = selected_global_indices_flat[1]

                profits_selected = beneficios_tensor_global[selected_city_indices, selected_item_local_indices]
                weights_selected = pesos_tensor_global[selected_city_indices, selected_item_local_indices]

                ratios_selected = torch.where(weights_selected > 0, profits_selected / weights_selected, torch.tensor(-1.0, device='cuda'))

                # Ordena todos los ítems seleccionados por ratio (ascendente, para eliminar los peores primero)
                sorted_ratios, sorted_indices_all_selected = torch.sort(ratios_selected, descending=False)

                num_candidates_to_check = min(sorted_ratios.numel(), TOP_CANDIDATES_FOR_ITEM_OPS)

                # Obtiene los índices reales (ciudad, ítem local) de los N peores candidatos
                candidate_indices_on_gpu = sorted_indices_all_selected[:num_candidates_to_check]

                # Trae a CPU solo los N candidatos para la selección aleatoria
                candidate_city_indices_cpu = selected_city_indices[candidate_indices_on_gpu].tolist()
                candidate_item_local_indices_cpu = selected_item_local_indices[candidate_indices_on_gpu].tolist()

                # Selecciona uno al azar de estos N candidatos (en CPU)
                chosen_candidate_idx = random.randrange(num_candidates_to_check)
                ciudad_idx_rem = candidate_city_indices_cpu[chosen_candidate_idx]
                item_local_idx_rem = candidate_item_local_indices_cpu[chosen_candidate_idx]

                nueva_mascara_gpu[ciudad_idx_rem, item_local_idx_rem] = False # Deselecciona el ítem

    # Randomly flip state of a couple of items (add/remove) - solo si hay ítems que voltear
    if torch.rand(1, device='cuda').item() < 0.5:
        # Get all existing items (not just selected)
        all_existing_items_flat_indices = torch.nonzero(mask_items_existentes, as_tuple=True)
        if all_existing_items_flat_indices[0].numel() > 0:
            num_existing_items = all_existing_items_flat_indices[0].numel()
            # Choose 2 random indices from all existing items
            random_indices_to_flip = torch.randint(0, num_existing_items, (2,), device='cuda')

            # Get the actual city and local item indices for these random picks
            city_indices_to_flip = all_existing_items_flat_indices[0][random_indices_to_flip]
            item_local_indices_to_flip = all_existing_items_flat_indices[1][random_indices_to_flip]

            for i in range(2):
                ciudad_to_flip_idx = city_indices_to_flip[i].item()
                item_to_flip_local_idx = item_local_indices_to_flip[i].item()
                # Toggle selection status directly on the GPU tensor
                nueva_mascara_gpu[ciudad_to_flip_idx, item_to_flip_local_idx] = \
                    ~nueva_mascara_gpu[ciudad_to_flip_idx, item_to_flip_local_idx]

    return nueva_ruta, nueva_mascara_gpu


# --- Ejecución Única de Simulated Annealing (MODIFICADO para máximo dinámico de ítems por ciudad y solución inicial) ---
def simulated_annealing_cuda(T0, alpha, Tmin, max_iter, datos_estructura, capacidad_t, v_min_t, v_max_t, costo_alq_t, solucion_inicial, max_items_ini, max_items_final, device='cuda'):
    (ciudades_ordenadas, items_por_ciudad_list_dummy, id_a_idx, idx_a_id,
     n_ciudades, total_items_dummy, dist_matrix, beneficios_tensor_global, pesos_tensor_global, mask_items_existentes,
     num_items_per_city_tensor, max_items_per_city_overall) = datos_estructura

    ruta_actual, seleccion_actual_gpu = solucion_inicial

    obj_actual, ganancia_actual, peso_actual, ratio_actual, velocidad_actual = \
        funcion_objetivo_cuda((ruta_actual, seleccion_actual_gpu), datos_estructura, capacidad_t, v_min_t, v_max_t, costo_alq_t, device=device)

    mejor_obj = obj_actual
    mejor_ruta = ruta_actual[:]
    mejor_seleccion_gpu = seleccion_actual_gpu.clone().detach()

    hist_obj = [obj_actual]
    hist_peso = [peso_actual]
    hist_ganancia = [ganancia_actual]
    hist_ratio = [ratio_actual]
    hist_vel = [velocidad_actual]

    no_change_counter = 0  # Contador de iteraciones sin cambios en la función objetivo

    T = T0

    for iteracion in range(max_iter):
        progreso = iteracion / max_iter
        max_items_dinamico = int(round(max_items_ini + (max_items_final - max_items_ini) * progreso))
        max_items_dinamico = max(1, min(INITIAL_MAX_ITEMS_PER_CITY, max_items_dinamico)) # Ensure bounds

        nueva_ruta, nueva_seleccion_gpu = vecino((ruta_actual, seleccion_actual_gpu), datos_estructura, capacidad_t, v_min_t, v_max_t, costo_alq_t, T, max_items_dinamico)
        obj_nuevo, ganancia_nueva, peso_nuevo, ratio_nuevo, vel_nueva = \
            funcion_objetivo_cuda((nueva_ruta, nueva_seleccion_gpu), datos_estructura, capacidad_t, v_min_t, v_max_t, costo_alq_t, device=device)
        delta = obj_nuevo - obj_actual

        if delta > 0 or (T > 1e-10 and random.random() < math.exp(delta / T)):
            # Se acepta la nueva solución
            ruta_actual = nueva_ruta
            seleccion_actual_gpu = nueva_seleccion_gpu
            if obj_nuevo == obj_actual:
                no_change_counter += 1
            else:
                no_change_counter = 0
            obj_actual = obj_nuevo
            ganancia_actual, peso_actual, ratio_actual, velocidad_actual = ganancia_nueva, peso_nuevo, ratio_nuevo, vel_nueva

            if obj_actual > mejor_obj:
                mejor_obj = obj_actual
                mejor_ruta = ruta_actual[:]
                mejor_seleccion_gpu = seleccion_actual_gpu.clone().detach()
        else:
            no_change_counter += 1

        hist_obj.append(obj_actual)
        hist_peso.append(peso_actual)
        hist_ganancia.append(ganancia_actual)
        hist_ratio.append(ratio_actual)
        hist_vel.append(velocidad_actual)

        if no_change_counter >= 200:
            break  # Salir del SA si 100 iteraciones consecutivas no cambian el objetivo

        T *= alpha
        if T < Tmin:
            break


    return mejor_obj, mejor_ruta, mejor_seleccion_gpu, hist_obj, hist_peso, hist_ganancia, hist_ratio, hist_vel

# --- Main Execution Block ---
if __name__ == '__main__':
    if not torch.cuda.is_available():
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] CUDA no está disponible. Ejecutando en CPU.")
        device = 'cpu'
    else:
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] CUDA está disponible. Ejecutando en GPU: {device_name}")
        device = 'cuda'

    FILE_PATH = 'a280_n279_uncorr_02.ttp' # 'rl1323_n13220_bounded-strongly-corr_05.ttp'
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Cargando problema TTP desde: {FILE_PATH}")
    items_data_raw, city_coords_raw, KNAP_CAP_total, MIN_SPEED, MAX_SPEED, RENT = parsear_ttp(FILE_PATH)

    # Se pasan los tensores globales creados en construir_estructuras_cuda
    (node_ids_sorted, items_per_city_indexed, id_to_idx, idx_to_id,
     num_cities, num_items_total, dist_matrix_gpu, beneficios_tensor_global, pesos_tensor_global,
     mask_items_existentes, num_items_per_city_tensor, max_items_per_city_overall) = \
        construir_estructuras_cuda(items_data_raw, city_coords_raw, device=device)

    # Convertir constantes a tensores GPU una sola vez
    KNAP_CAP_total_t = torch.tensor(KNAP_CAP_total, device=device, dtype=torch.float32)
    MIN_SPEED_t = torch.tensor(MIN_SPEED, device=device, dtype=torch.float32)
    MAX_SPEED_t = torch.tensor(MAX_SPEED, device=device, dtype=torch.float32)
    RENT_t = torch.tensor(RENT, device=device, dtype=torch.float32)

    # Agrupar datos estructurales del problema para pasarlos como un solo argumento
    problem_data_structure = (
        node_ids_sorted, items_per_city_indexed, id_to_idx, idx_to_id,
        num_cities, num_items_total, dist_matrix_gpu, beneficios_tensor_global, pesos_tensor_global,
        mask_items_existentes, num_items_per_city_tensor, max_items_per_city_overall
    )

    # SA Parameters for single-solution run
    T0 = 200
    alpha = 0.999
    Tmin = 0.45
    max_iters = 700 # Iteraciones para cada mini-SA
    INITIAL_MAX_ITEMS_PER_CITY = 5
    FINAL_MAX_ITEMS_PER_CITY = 1

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Generando solución inicial...")
    current_tour = insercion_mas_lejano_dp_cuda(dist_matrix_gpu)
    current_mask_gpu, _ = greedy_mochila_inicial_por_ciudad_cuda(
        items_per_city_indexed, KNAP_CAP_total, INITIAL_MAX_ITEMS_PER_CITY,
        beneficios_tensor_global, pesos_tensor_global, mask_items_existentes, device=device
    )

    current_solution = (current_tour, current_mask_gpu)
    # Evaluate initial solution
    initial_obj, _, _, _, _ = funcion_objetivo_cuda(current_solution, problem_data_structure, KNAP_CAP_total_t, MIN_SPEED_t, MAX_SPEED_t, RENT_t, device=device)
    best_obj = initial_obj
    overall_solution = current_solution # (best_tour, best_mask_gpu)

    NUM_GENERACIONES = 200   # Unificación: número de generaciones para cada simulación

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Iniciando simulación con {NUM_GENERACIONES} generaciones.")
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Mejor objetivo inicial: {best_obj:.2f}")

    for gen in range(NUM_GENERACIONES):
        new_obj, new_tour, new_mask_gpu, _, _, _, _, _ = simulated_annealing_cuda(
            T0, alpha, Tmin, max_iters, problem_data_structure, KNAP_CAP_total_t, MIN_SPEED_t, MAX_SPEED_t, RENT_t, current_solution,
            INITIAL_MAX_ITEMS_PER_CITY, FINAL_MAX_ITEMS_PER_CITY, device=device
        )
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Generación {gen+1}/{NUM_GENERACIONES}: Obj = {new_obj:.2f}")
        current_solution = (new_tour, new_mask_gpu)
        if new_obj > best_obj:
            best_obj = new_obj
            overall_solution = current_solution

    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Mejor Objetivo Global Encontrado: {best_obj:.2f}")

    # --- Experimentos de Simulación ---
    num_simulaciones = 10   # Se realizarán 10 simulaciones en total
    final_obj_simulations = []  # Se guarda el objetivo final de cada experimento

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Iniciando experimento: {num_simulaciones} simulaciones, cada una con {NUM_GENERACIONES} generaciones.")

    for sim in range(num_simulaciones):
        # Crear solución inicial aleatoria para esta simulación
        initial_tour = insercion_mas_lejano_dp_cuda(dist_matrix_gpu)
        initial_mask, _ = greedy_mochila_inicial_por_ciudad_cuda(
            items_per_city_indexed, KNAP_CAP_total, INITIAL_MAX_ITEMS_PER_CITY,
            beneficios_tensor_global, pesos_tensor_global, mask_items_existentes, device=device
        )
        current_solution_sim = (initial_tour, initial_mask)
        best_obj_sim = -float('inf')
        # Ejecutar SA por NUM_GENERACIONES iteraciones
        for gen in range(NUM_GENERACIONES):
             new_obj, new_tour, new_mask_gpu, _, _, _, _, _ = simulated_annealing_cuda(
                    T0, alpha, Tmin, max_iters, problem_data_structure, KNAP_CAP_total_t, MIN_SPEED_t, MAX_SPEED_t, RENT_t,
                    current_solution_sim, INITIAL_MAX_ITEMS_PER_CITY, FINAL_MAX_ITEMS_PER_CITY, device=device
             )
             current_solution_sim = (new_tour, new_mask_gpu)
             if new_obj > best_obj_sim:
                    best_obj_sim = new_obj
        final_obj_simulations.append(best_obj_sim)
        print(f"Simulación {sim+1}: Mejor Obj = {best_obj_sim:.2f}")

    # Calcular estadísticas sobre los 10 resultados finales
    media = np.mean(final_obj_simulations)
    mediana = np.median(final_obj_simulations)
    std_dev = np.std(final_obj_simulations)
    skew_val = skew(final_obj_simulations)
    kurtosis_val = kurtosis(final_obj_simulations)

    print("\nResumen estadístico:")
    print(f"Media: {media:.4f}")
    print(f"Mediana: {mediana:.4f}")
    print(f"Desviación Estándar: {std_dev:.4f}")
    print(f"Skew: {skew_val:.4f}")
    print(f"Curtosis: {kurtosis_val:.4f}")

    # Graficar: un histograma y un boxplot de los 10 mejores objetivos
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(final_obj_simulations, bins=5, color='skyblue', edgecolor='black')
    plt.title('Histograma de Mejor Objetivo por Simulación')
    plt.xlabel('Valor Objetivo')
    plt.ylabel('Frecuencia')

    plt.subplot(1,2,2)
    plt.boxplot(final_obj_simulations, showfliers=False)
    plt.title('Boxplot de Mejor Objetivo por Simulación')
    plt.ylabel('Valor Objetivo')

    plt.tight_layout()
    plt.show()