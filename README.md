# Tarea 2 Optimización

**Integrantes:**  
* Patricio Figueroa  
* Kavon Kermani  
* Gabriel Sanzana

---

## Nota importante sobre la ejecución del código

Este código **requiere ejecución local con soporte CUDA habilitado** para funcionar correctamente.  
Google Colab **no permite el uso directo de CUDA** desde entornos virtuales personalizados, por lo tanto, **el código no funcionará correctamente en Colab**, incluso si se activa la GPU.

Además, **la instancia a probar debe ser especificada manualmente dentro del código**, modificando directamente el nombre del archivo de instancia en `CodigoTTP.py`, `Trayectoria_solucion.py` y `Graficos_histograma_boxplot.py`.

Para **visualizar los gráficos**, se deben ejecutar manualmente los siguientes scripts:

- `Trayectoria_solucion.py`: genera múltiples gráficos relacionados con la solución final, incluyendo la ruta seguida, beneficio acumulado, peso, velocidad, ratio beneficio/peso, distancia recorrida y función objetivo.
- `Graficos_histograma_boxplot.py`: genera histogramas y boxplots relacionados con el desempeño del algoritmo.

---

## Demostración de la ejecución del Código

Video demostrativo del funcionamiento del código:  
🔗 [Ver video](https://drive.google.com/file/d/1kqfbeK3RSB6sGN81tZ9JD1eg4siec9-z/view?usp=sharing)

---

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instalado:

- Python 3.8 o superior  
- NVIDIA GPU compatible con CUDA  
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)  
- [PyTorch con soporte CUDA](https://pytorch.org/get-started/locally/)  
- Bibliotecas: `Numba`, `Torch`.

---

## Ejecución del código

1. Clona este repositorio o descarga los archivos del proyecto.

2. Abre una terminal y activa tu entorno virtual:

```bash
# Activar entorno virtual (Windows)
.venv\Scripts\activate

# Ejecutar script principal
python CodigoTTP.py
