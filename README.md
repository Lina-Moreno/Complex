# Complex — Cifrado experimental de imágenes

Prototipo académico desarrollado en el contexto de **Complex UD** para explorar el cifrado y descifrado de imágenes usando **Python**, transformaciones con **matrices de Pauli**, secuencias del **mapa logístico** y **autómatas celulares**.

## Objetivo

Estudiar un flujo reversible de transformación de imágenes RGB y analizar el comportamiento de las claves, los parámetros y las imágenes resultantes.

## Funcionamiento general

El módulo `RevisionEncrypt/pipeline.py` organiza estas etapas:

1. **Preparación:** lectura de la imagen y separación de sus canales RGB.
2. **Generación de parámetros:** obtención de claves y secuencias a partir del mapa logístico.
3. **Difusión:** transformación de bloques de datos de imagen mediante matrices construidas con operadores de Pauli.
4. **Representación binaria:** tratamiento de componentes de los canales y conversión a bits.
5. **Permutación:** transformación de bloques de 8 bits mediante reglas de autómatas celulares.
6. **Descifrado:** aplicación de transformaciones inversas y reconstrucción de los canales.

```text
Imagen RGB
   -> parámetros y clave
   -> difusión con matrices de Pauli
   -> representación binaria
   -> autómatas celulares
   -> datos transformados
   -> operaciones inversas
   -> imagen reconstruida
```

## Tecnologías

- Python y Jupyter Notebook
- NumPy para matrices y procesamiento numérico
- Pillow (PIL) para lectura y manipulación de imágenes
- Mapas logísticos, operadores de Pauli y autómatas celulares

## Estructura del proyecto

```text
Complex/
├── Encriptacion/
│   └── encrypt.ipynb
├── Desencriptacion/
│   └── decrypt.ipynb
├── Pruebas/
│   └── analisis_estadistico.ipynb
├── Imagenes/
├── Resultados/
└── RevisionEncrypt/
    ├── pipeline.py
    ├── caos.py
    ├── difusion_pauli.py
    ├── permutacion_automata.py
    ├── io_imagen.py
    ├── metricas.py
    ├── encrypt.ipynb
    ├── decrypt.ipynb
    └── pruebas_2.ipynb
```

## Cómo explorar el proyecto

**Requisitos básicos:** Python 3, Jupyter, NumPy y Pillow. Algunos notebooks podrían requerir bibliotecas adicionales.

```bash
git clone https://github.com/Lina-Moreno/Complex.git
cd Complex
python -m venv .venv
```

Activa el entorno virtual:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
```

```bash
pip install numpy pillow jupyter matplotlib
jupyter notebook
```

Abre los notebooks de `RevisionEncrypt/` para explorar el flujo de cifrado, descifrado y pruebas. **Antes de ejecutarlos, verifica y ajusta las rutas de imágenes y archivos usadas por cada notebook.** Este repositorio todavía no dispone de un proceso de instalación y ejecución único verificado.

## Pruebas y resultados

El repositorio contiene notebooks para pruebas y análisis de imágenes. Para documentar resultados reproducibles, registra el archivo de entrada, sus dimensiones, los parámetros usados, el resultado cifrado, la reconstrucción y las métricas correspondientes.
