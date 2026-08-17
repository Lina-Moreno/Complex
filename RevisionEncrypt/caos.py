"""
Mapa logistico caotico y derivacion de la llave de cifrado.

La llave (lista de indices 1-24 que seleccionan matrices de Pauli), la
semilla x0 y el numero de iteraciones ya consumidas se derivan del
contenido de la imagen (promedio de su primera fila). La misma secuencia
del mapa logistico, continuada desde `iteraciones_usadas`, se usa despues
en permutacion_automata.generar_indices_reglas para elegir las reglas de
cada fila.
"""
import numpy as np


def generar_con_logistico(x0, n, iteraciones_ignorar: int = 0, r: float = 3.99999):
    """Itera x_{n+1} = r*x_n*(1-x_n), descarta 'iteraciones_ignorar' pasos y
    devuelve los siguientes n valores."""
    x = x0
    for _ in range(iteraciones_ignorar):
        x = r * x * (1 - x)
    valores = []
    for _ in range(n):
        x = r * x * (1 - x)
        valores.append(x)
    return valores


def generar_llave(imagen_R, imagen_G, imagen_B, verbose=False, epsilon_x0: float = 0.0):
    """Deriva (llave, x0, iteraciones_usadas) a partir del promedio de la
    primera fila de la imagen.

    epsilon_x0 permite perturbar la semilla de forma controlada para el
    analisis de sensibilidad (efecto avalancha). Con epsilon_x0=0.0 (valor
    por defecto) el resultado es identico al usado en el cifrado.
    """
    primer_fila_R = np.array(imagen_R)[0]
    primer_fila_G = np.array(imagen_G)[0]
    primer_fila_B = np.array(imagen_B)[0]

    primera_fila = []
    for i in range(len(primer_fila_R)):
        primera_fila.append(primer_fila_R[i])
        primera_fila.append(primer_fila_G[i])
        primera_fila.append(primer_fila_B[i])

    promedio = np.mean(primera_fila)
    parte_entera = int(promedio)               # iteraciones a ignorar
    K = sum(int(d) for d in str(parte_entera))  # cantidad de valores a generar

    x0 = promedio / 1000 + epsilon_x0
    secuencia_caotica_llave = generar_con_logistico(x0, K, parte_entera)

    secuencia = []
    for x in secuencia_caotica_llave:
        x_abs = abs(x)
        parte_entera_x = int(x_abs)
        parte_fraccionaria = x_abs - parte_entera_x
        grande = int(parte_fraccionaria * 10**14)
        numero = grande % 24
        secuencia.append(numero + 1)

    posicion_medio = (K + 1) // 2 - 1 if K % 2 == 1 else K // 2 - 1
    valor_central = secuencia[posicion_medio]
    ultimo_bit = valor_central % 2

    if ultimo_bit == 0:
        matrices_finales = [secuencia[i] for i in range(posicion_medio)]
    else:
        matrices_finales = [secuencia[i] for i in range(posicion_medio + 1, K)]

    iteraciones_usadas = parte_entera + K

    if verbose:
        print(f"Promedio primera fila: {promedio:.4f}")
        print(f"Parte entera (iteradas a ignorar): {parte_entera}")
        print(f"x0 (semilla): {x0}")
        print(f"K (valores a generar): {K}")
        print(f"Secuencia completa: {secuencia}")
        print(f"Posicion media: {posicion_medio}   Valor central: {valor_central}")
        direccion = "derecha" if ultimo_bit == 1 else "izquierda"
        print(f"Ultimo bit: {ultimo_bit}  ->  direccion: {direccion}")
        print(f"LLAVE GENERADA: {matrices_finales}  (longitud={len(matrices_finales)})")

    return matrices_finales, x0, iteraciones_usadas
