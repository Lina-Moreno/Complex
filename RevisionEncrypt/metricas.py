"""
Metricas para evaluar el cifrado: error de reconstruccion (MSE/PSNR) y
sensibilidad a la llave (NPCR/UACI).
"""
import math
import numpy as np


def calcular_mse(img1, img2):
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)


def calcular_psnr(img1, img2):
    mse = calcular_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def npcr(c1, c2):
    """Number of Pixels Change Rate: % de pixeles distintos entre dos
    imagenes."""
    return 100.0 * np.mean(c1.astype(np.int64) != c2.astype(np.int64))


def uaci(c1, c2):
    """Unified Average Changing Intensity: diferencia promedio de
    intensidad entre dos imagenes."""
    return 100.0 * np.mean(np.abs(c1.astype(np.float64) - c2.astype(np.float64)) / 255.0)
