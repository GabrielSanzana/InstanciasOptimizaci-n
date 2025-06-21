# Tarea 2 Optimización

Integrantes: 
* Patricio Figueroa
* Kavon Kermani
* Gabriel Sanzana

##Nota importante sobre la ejecución del código

Este código requiere ejecución local con soporte CUDA habilitado para funcionar correctamente.
Google Colab no permite el uso directo de CUDA desde entornos virtuales personalizados, por lo tanto, este código no funcionará correctamente en Colab, incluso si se activa la GPU

---

## Demostración de la ejecución del Código

Video demostrativo del funcionamiento del código: https://drive.google.com/file/d/1kqfbeK3RSB6sGN81tZ9JD1eg4siec9-z/view?usp=sharing

##  Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instalado:

- Python 3.8 o superior
- NVIDIA GPU compatible con CUDA
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [PyTorch con soporte CUDA](https://pytorch.org/get-started/locally/)
- `Numba`, `NumPy`, y otras dependencias

## Ejecución del código

- Clona este repositorio o descarga los archivos del proyecto.

- Abre una terminal y activa tu entorno virtual:

```bash
# Activar entorno virtual
.venv/Scripts/activate

# Ejecutar script principal
python TTP.py
