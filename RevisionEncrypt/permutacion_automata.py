"""
Etapa de permutacion: automata celular por bloques de 8 bits.

Cada fila (ya convertida a bits) se divide en bloques de 8 bits. Para cada
fila se elige, a partir de la misma secuencia caotica usada para la llave,
uno de los 16 conjuntos de reglas de TABLA2. Cada conjunto trae las 8
reglas para cifrar (indice 0) y las 8 reglas inversas para descifrar
(indice 1), una por posicion dentro del bloque.
"""
import numpy as np
from caos import generar_con_logistico


def regla_A(p, q, r):   return p          # 240: shift-right
def regla_C(p, q, r):   return r          # 170: shift-left
def regla_B(p, q, r):   return q          # 204: identidad
def regla_Ab(p, q, r):  return 1 - p      # 15:  shift-right + complemento
def regla_Cb(p, q, r):  return 1 - r      # 85:  shift-left + complemento
def regla_Bb(p, q, r):  return 1 - q      # 51:  complemento

REGLAS = {
    'A': regla_A, 'C': regla_C, 'B': regla_B,
    'Ab': regla_Ab, 'Cb': regla_Cb, 'Bb': regla_Bb
}

TABLA2 = {
    0:  (['A','A','A','Ab','Ab','A','Ab','Ab'],   ['C','C','Cb','Cb','C','Cb','Cb','C']),
    1:  (['A','Ab','Ab','A','Ab','Ab','A','A'],   ['Cb','Cb','C','Cb','Cb','C','C','C']),
    2:  (['Ab','A','Ab','Ab','A','A','A','Ab'],   ['C','Cb','Cb','C','C','C','Cb','Cb']),
    3:  (['Ab','Ab','A','A','A','Ab','Ab','A'],   ['Cb','C','C','C','Cb','Cb','C','Cb']),
    4:  (['C','C','Cb','C','C','Cb','Cb','Cb'],   ['Ab','A','A','Ab','A','A','Ab','Ab']),
    5:  (['Cb','Cb','C','C','Cb','C','C','Cb'],   ['Ab','Ab','Ab','A','A','Ab','A','A']),
    6:  (['C','Cb','Cb','Cb','C','C','Cb','C'],   ['A','A','Ab','Ab','Ab','A','A','Ab']),
    7:  (['Cb','C','C','Cb','Cb','Cb','C','C'],   ['A','Ab','A','A','Ab','Ab','Ab','A']),
    8:  (['B','B','Bb','B','C','Ab','C','Ab'],    ['B','B','Bb','B','Cb','A','Cb','A']),
    9:  (['Bb','Cb','A','B','Cb','A','B','Bb'],   ['Bb','C','Ab','B','C','Ab','B','Bb']),
    10: (['C','Ab','Bb','Bb','B','C','Ab','B'],   ['Cb','A','Bb','Bb','B','Cb','A','B']),
    11: (['Bb','B','C','Ab','Bb','Bb','C','A'],   ['Bb','B','Cb','A','Bb','Bb','C','A']),
    12: (['Bb','Cb','A','C','A','Bb','Bb','B'],   ['Bb','C','Ab','C','A','Bb','Bb','B']),
    13: (['Bb','B','Bb','Cb','A','C','A','Bb'],   ['Bb','B','Bb','C','Ab','C','A','Bb']),
    14: (['C','A','B','Cb','Ab','B','Bb','Bb'],   ['C','A','B','Cb','Ab','B','Bb','Bb']),
    15: (['B','Bb','Cb','A','Bb','Cb','A','B'],   ['B','Bb','C','Ab','Bb','C','Ab','B']),
}


def evolucionar_bloque(bloque, reglas_8):
    """Aplica, a un bloque circular de 8 bits, la regla correspondiente a
    cada posicion (reglas_8[i]), leyendo los vecinos izquierdo y derecho
    dentro del mismo bloque (frontera circular de longitud 8)."""
    longitud = len(bloque)
    nuevo = np.zeros(longitud, dtype=np.uint8)
    for i in range(longitud):
        izq = bloque[(i - 1) % longitud]
        cen = bloque[i]
        der = bloque[(i + 1) % longitud]
        nuevo[i] = REGLAS[reglas_8[i]](izq, cen, der)
    return nuevo


def evolucionar_fila(fila_bits, reglas_8):
    """Divide la fila en bloques de 8 bits y evoluciona cada uno por
    separado con el mismo conjunto de 8 reglas."""
    r = 8
    n = len(fila_bits)
    assert n % r == 0, f"La fila debe ser multiplo de {r} (revisa el padding)"

    nueva_fila = np.zeros(n, dtype=np.uint8)
    for inicio in range(0, n, r):
        bloque = fila_bits[inicio:inicio + r]
        nueva_fila[inicio:inicio + r] = evolucionar_bloque(bloque, reglas_8)
    return nueva_fila


def generar_indices_reglas(x0, n_filas, iteraciones_ignorar, r=3.99999):
    """Para cada fila de la imagen, deriva un indice 0-15 (que conjunto de
    TABLA2 usar) a partir del mapa logistico."""
    valores = generar_con_logistico(x0, n_filas, iteraciones_ignorar, r)
    reglas_idx = []
    for v in valores:
        v_abs = abs(v)
        frac = v_abs - int(v_abs)
        grande = int(frac * 10**14)
        reglas_idx.append(grande % 16)
    return reglas_idx


def cifrar_filas_automata(filas_bin, reglas_idx):
    """Cifra cada fila binaria con el conjunto de reglas directo
    (TABLA2[idx][0])."""
    filas_resultado = []
    for fila, idx in zip(filas_bin, reglas_idx):
        reglas_8 = TABLA2[idx][0]
        filas_resultado.append(evolucionar_fila(fila, reglas_8))
    return filas_resultado


def descifrar_canal_ca(filas_cif, reglas_idx):
    """Inversa de cifrar_filas_automata: usa el conjunto de reglas inverso
    (TABLA2[idx][1])."""
    filas_resultado = []
    for fila, idx in zip(filas_cif, reglas_idx):
        reglas_8 = TABLA2[idx][1]
        filas_resultado.append(evolucionar_fila(fila, reglas_8))
    return filas_resultado
