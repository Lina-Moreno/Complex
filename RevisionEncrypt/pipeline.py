"""
Orquesta el cifrado de Encriptacion/encrypt.ipynb (difusion Pauli y automata
celular sobre R, G y B intercalados) y su inverso para los notebooks de
analisis.
"""
from dataclasses import dataclass

import numpy as np

from caos import generar_llave
from difusion_pauli import (
    generar_matrices_4x4, matrices_encriptacion, obtener_rgb_como_bloques,
    cifrar_imagen, descifrar_canal_pauli, bloques_a_canal, canal_a_2d,
    canal_2d_a_bloques, separar_en_tres_capas, reconstruir_complejo,
)
from permutacion_automata import (
    generar_indices_reglas, cifrar_filas_automata, descifrar_canal_ca,
)
from io_imagen import canales_a_filas_binarias, filas_a_matrices_rgb


@dataclass
class ImagenCifrada:
    """Resultado de cifrar_imagen_completa. Guarda exactamente lo que hace
    falta para descifrar despues (y es, campo a campo, lo que encrypt.ipynb
    escribe en el TIFF + cifrado_meta.txt).

    filas_RGB_cif contiene los bits R, G y B intercalados por pixel, con
    padding a multiplo de 8, igual que la pagina 1 del TIFF.
    """
    filas_RGB_cif: list
    num_bits: int
    key: list
    x0: float
    iteraciones_usadas: int
    r_imag: np.ndarray
    r_signos: np.ndarray
    g_imag: np.ndarray
    g_signos: np.ndarray
    b_imag: np.ndarray
    b_signos: np.ndarray
    h: int
    w: int


def cifrar_imagen_completa(r_img, g_img, b_img, verbose=False):
    """Ejecuta difusion (Pauli) + permutacion (automata celular) sobre los
    3 canales de una imagen y devuelve un ImagenCifrada con todo lo
    necesario para descifrarla despues."""
    h, w = r_img.height, r_img.width

    key, x0, iteraciones_usadas = generar_llave(r_img, g_img, b_img, verbose=verbose)

    matrices_rotadas = generar_matrices_4x4()
    encriptacion = matrices_encriptacion(key, matrices_rotacion=matrices_rotadas)

    R_b, G_b, B_b = obtener_rgb_como_bloques(r_img, g_img, b_img)
    c_r, c_g, c_b = cifrar_imagen(R_b, G_b, B_b, matrices_finales=encriptacion)

    c_r_2d = canal_a_2d(c_r, h, w)
    c_g_2d = canal_a_2d(c_g, h, w)
    c_b_2d = canal_a_2d(c_b, h, w)

    r_real, r_imag, r_signos = separar_en_tres_capas(c_r_2d)
    g_real, g_imag, g_signos = separar_en_tres_capas(c_g_2d)
    b_real, b_imag, b_signos = separar_en_tres_capas(c_b_2d)

    max_val = int(np.max([r_real.max(), g_real.max(), b_real.max()]))
    num_bits = max_val.bit_length()

    filas_RGB = canales_a_filas_binarias(
        r_real.astype(int), g_real.astype(int), b_real.astype(int), num_bits
    )

    reglas = generar_indices_reglas(x0, h, iteraciones_ignorar=iteraciones_usadas)

    filas_RGB_cif = cifrar_filas_automata(filas_RGB, reglas)

    return ImagenCifrada(
        filas_RGB_cif=filas_RGB_cif,
        num_bits=num_bits, key=key, x0=x0, iteraciones_usadas=iteraciones_usadas,
        r_imag=r_imag, r_signos=r_signos, g_imag=g_imag, g_signos=g_signos,
        b_imag=b_imag, b_signos=b_signos, h=h, w=w,
    )


def descifrar_imagen_completa(cifrado: ImagenCifrada, key=None, x0=None, iteraciones_usadas=None):
    """Inversa de cifrar_imagen_completa: deshace primero el automata
    celular y despues Pauli.

    key/x0/iteraciones_usadas son opcionales y sirven para el analisis de
    sensibilidad: si no se pasan, se usan los valores correctos guardados
    en 'cifrado' (descifrado normal). Si se pasa un x0 distinto, tambien
    cambian las reglas del automata que se usan para descifrar, porque
    generar_indices_reglas depende de x0 — igual que le pasaria a un
    atacante que solo conoce una aproximacion de la semilla real."""
    key = cifrado.key if key is None else key
    x0 = cifrado.x0 if x0 is None else x0
    iteraciones_usadas = cifrado.iteraciones_usadas if iteraciones_usadas is None else iteraciones_usadas

    reglas = generar_indices_reglas(x0, cifrado.h, iteraciones_ignorar=iteraciones_usadas)

    filas_RGB = descifrar_canal_ca(cifrado.filas_RGB_cif, reglas)
    R_int, G_int, B_int = filas_a_matrices_rgb(filas_RGB, cifrado.num_bits, cifrado.w)

    c_r_2d = reconstruir_complejo(R_int, cifrado.r_imag, cifrado.r_signos)
    c_g_2d = reconstruir_complejo(G_int, cifrado.g_imag, cifrado.g_signos)
    c_b_2d = reconstruir_complejo(B_int, cifrado.b_imag, cifrado.b_signos)

    bloques_r = canal_2d_a_bloques(c_r_2d)
    bloques_g = canal_2d_a_bloques(c_g_2d)
    bloques_b = canal_2d_a_bloques(c_b_2d)

    matrices_rotadas = generar_matrices_4x4()
    encriptacion = matrices_encriptacion(key, matrices_rotacion=matrices_rotadas)

    p_r = descifrar_canal_pauli(bloques_r, encriptacion)
    p_g = descifrar_canal_pauli(bloques_g, encriptacion)
    p_b = descifrar_canal_pauli(bloques_b, encriptacion)

    R_dec = bloques_a_canal(p_r, cifrado.h, cifrado.w)
    G_dec = bloques_a_canal(p_g, cifrado.h, cifrado.w)
    B_dec = bloques_a_canal(p_b, cifrado.h, cifrado.w)


    return np.stack([R_dec, G_dec, B_dec], axis=-1)
