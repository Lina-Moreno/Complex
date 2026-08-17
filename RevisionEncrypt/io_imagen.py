"""
Lectura/escritura de imagenes, conversion canal <-> filas de bits, y
persistencia de los metadatos de cifrado (llave, x0, iteraciones, num_bits).
"""
import numpy as np
from PIL import Image


def obtener_rgb(ruta_png):
    img = Image.open(ruta_png).convert("RGB")
    r_img, g_img, b_img = img.split()
    return r_img, g_img, b_img


def visualizar_canal_color(canal_img, color):
    color = color.upper()
    if color not in ["R", "G", "B"]:
        raise ValueError("color debe ser 'R', 'G' o 'B'")

    arr = np.array(canal_img)
    zeros = np.zeros_like(arr)

    if color == "R":
        rgb = np.stack([arr, zeros, zeros], axis=2)
    elif color == "G":
        rgb = np.stack([zeros, arr, zeros], axis=2)
    else:
        rgb = np.stack([zeros, zeros, arr], axis=2)

    return Image.fromarray(rgb, mode="RGB")


def canal_a_filas_binarias(canal_int, n_bits):
    """Convierte cada fila de pixeles (enteros) en una cadena de bits de
    longitud w*n_bits, rellenada con ceros hasta el siguiente multiplo de 8
    para que evolucionar_fila pueda dividirla en bloques de 8 bits."""
    filas_binarias = []
    for fila in canal_int:
        lista_bits_por_pixel = [format(int(valor), f'0{n_bits}b') for valor in fila]
        cadena_bits = ''.join(lista_bits_por_pixel)

        residuo = len(cadena_bits) % 8
        cantidad_ceros_extra = 8 - residuo if residuo != 0 else 0
        cadena_bits_final = cadena_bits + ('0' * cantidad_ceros_extra)

        fila_np = np.array([int(c) for c in cadena_bits_final], dtype=np.uint8)
        filas_binarias.append(fila_np)
    return filas_binarias


def filas_a_matriz_pixeles(filas, n_bits, w):
    """Inversa de canal_a_filas_binarias: descarta el padding y reagrupa
    los bits en pixeles de n_bits."""
    h = len(filas)
    matriz = np.zeros((h, w), dtype=np.int64)
    for i, fila in enumerate(filas):
        util = fila[:w * n_bits]
        for j in range(w):
            bloque = util[j * n_bits:(j + 1) * n_bits]
            matriz[i, j] = int(''.join(str(b) for b in bloque), 2)
    return matriz


def normalizar(arr):
    """Normaliza a uint8 [0,255] solo para poder visualizar un canal
    cifrado (sus valores no estan en rango de pixel)."""
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def leer_metadatos(ruta_meta):
    datos = {}
    with open(str(ruta_meta), "r") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or "=" not in linea:
                continue
            clave, valor = linea.split("=", 1)
            datos[clave.strip()] = valor.strip()
    return datos


def bits_pixel(total_bits_fila, w, rango=(1, 32)):
    """Heuristica de compatibilidad: reconstruye num_bits a partir del
    ancho (con padding) de una fila cifrada, para archivos generados antes
    de que 'num_bits' se guardara en cifrado_meta.txt. Si el archivo tiene
    ese campo, usarlo directamente en vez de esta funcion."""
    posibles_bits = []
    for k in range(total_bits_fila - 7, total_bits_fila + 1):
        if k > 0 and k % w == 0:
            nb = k // w
            if rango[0] <= nb <= rango[1]:
                posibles_bits.append(nb)
    return posibles_bits
