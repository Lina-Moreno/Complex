"""
Etapa de difusion: cifrado con matrices de Pauli.

Cada canal se agrupa en bloques de 4 valores. Se generan las 24
permutaciones de {I, sigma_x, sigma_y, sigma_z} en las 4 posiciones de una
matriz 4x4 (dos matrices 2x2 apiladas por fila), y la llave selecciona,
para cada posicion de la llave, cual de esas 24 matrices se aplica al
bloque (multiplicacion matriz-bloque en cadena).
"""
import numpy as np
from itertools import permutations


def generar_matrices_4x4():
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    ops = [I, sigma_x, sigma_y, sigma_z]

    matrices_4x4 = []
    for M1, M2, M3, M4 in permutations(ops, 4):
        M = np.block([[M1, M2], [M3, M4]])
        matrices_4x4.append(M)
    return matrices_4x4


def matrices_encriptacion(llave, matrices_rotacion):
    """Selecciona, para cada elemento de la llave (1-24), la matriz 4x4
    correspondiente."""
    return [matrices_rotacion[(item - 1) % 24] for item in llave]


def obtener_rgb_como_bloques(R, G, B):
    """Convierte cada canal (imagen PIL en escala de grises) en una matriz
    de bloques (4, N), aplanando la imagen y agrupando de a 4 valores."""
    def canal_a_bloques(canal_img):
        arr = np.array(canal_img).flatten().astype(float)
        return arr.reshape(4, -1)
    return canal_a_bloques(R), canal_a_bloques(G), canal_a_bloques(B)


def cifrar_canal_pauli(canal, matrices_finales):
    """Multiplica el canal (bloques 4xN, tipo complejo) por cada matriz de
    la llave, en orden."""
    resultado = canal.astype(complex)
    for Mi in matrices_finales:
        resultado = Mi @ resultado
    return resultado


def cifrar_imagen(r, g, b, matrices_finales):
    Cr = cifrar_canal_pauli(r, matrices_finales)
    Cg = cifrar_canal_pauli(g, matrices_finales)
    Cb = cifrar_canal_pauli(b, matrices_finales)
    return Cr, Cg, Cb


def descifrar_canal_pauli(bloque_cif, matrices_finales):
    """Inversa de cifrar_canal_pauli: aplica las matrices inversas en orden
    contrario."""
    resultado = bloque_cif.astype(complex)
    for Mi in reversed(matrices_finales):
        resultado = np.linalg.inv(Mi) @ resultado
    return resultado


def bloques_a_canal(bloque_plano, h, w):
    """Convierte bloques (4, N) descifrados de vuelta a una imagen (h, w)
    de 8 bits: toma la parte real, redondea y recorta a [0, 255].

    Esta es la unica funcion que debe usarse para pasar de bloques
    descifrados a imagen; evita reimplementar la conversion (con
    np.abs u otra variante) en cada notebook o funcion de prueba."""
    real = np.round(bloque_plano.real).astype(np.int64)
    real = np.clip(real, 0, 255)
    return real.flatten().reshape(h, w).astype(np.uint8)


def canal_a_2d(canal_cifrado, h, w):
    return canal_cifrado.flatten().reshape(h, w)


def canal_2d_a_bloques(canal_2d):
    return canal_2d.flatten().reshape(4, -1)


def separar_en_tres_capas(canal_2d_complejo):
    """Separa un canal complejo (h, w) en magnitud real, magnitud
    imaginaria y un mapa de signos (0..3), para poder guardar cada parte
    como una capa sin numeros negativos ni complejos."""
    real_original = canal_2d_complejo.real
    imag_original = canal_2d_complejo.imag

    real_abs = np.abs(real_original)
    imag_abs = np.abs(imag_original)

    real_negativo = real_original < 0
    imag_negativo = imag_original < 0

    # Signos: 0 (+,+), 1 (-,+), 2 (+,-), 3 (-,-)
    mapa_signos = np.zeros_like(real_original, dtype=np.float64)
    mapa_signos[real_negativo & ~imag_negativo] = 1.0
    mapa_signos[~real_negativo & imag_negativo] = 2.0
    mapa_signos[real_negativo & imag_negativo] = 3.0

    return real_abs, imag_abs, mapa_signos


def reconstruir_complejo(mag_real, mag_imag, signos):
    """Inversa de separar_en_tres_capas."""
    signo_real = np.where(np.isin(signos, [1, 3]), -1, 1)
    signo_imag = np.where(np.isin(signos, [2, 3]), -1, 1)
    return (mag_real * signo_real) + 1j * (mag_imag * signo_imag)


def descifrar_rgb(Cr, Cg, Cb, llave, h, w, matrices_rotadas):
    """Descifra los 3 canales (ya en bloques 4xN) con una llave dada y arma
    la imagen RGB (h, w, 3). Util para comparar, en el analisis de
    sensibilidad, el mismo texto cifrado descifrado con llaves distintas."""
    enc = matrices_encriptacion(llave, matrices_rotadas)
    Pr = descifrar_canal_pauli(Cr, enc)
    Pg = descifrar_canal_pauli(Cg, enc)
    Pb = descifrar_canal_pauli(Cb, enc)
    Rc = bloques_a_canal(Pr, h, w)
    Gc = bloques_a_canal(Pg, h, w)
    Bc = bloques_a_canal(Pb, h, w)
    return np.stack([Rc, Gc, Bc], axis=-1)
