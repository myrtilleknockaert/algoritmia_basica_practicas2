# 📌 Practica 2: Programacion Dinamica

## 📌 1. Presentación de la práctica
Este proyecto implementa una **reducción de costuras (seam carving)** para el procesamiento de imágenes.  
El objetivo es **reducir el ancho de las imágenes** mediante la eliminación iterativa de la costura de menor energía utilizando **programación dinámica**.

## 📌 2. Características
* Reducción del ancho de la imagen mediante la eliminación de costuras.
* Cálculo de la energía de cada píxel basado en la suma de los valores RGB.
* Uso de programación dinámica para determinar la costura (camino) de menor energía.
* Visualización de imágenes intermedias y finales.
* Soporte para eliminar múltiples costuras de forma iterativa.

---

## 📌 3. Organización de archivos
### 📂 `practica2_<NIA>`
El directorio contiene:
- **📜 `README.md`** → Explicación del proyecto (este archivo).
- **📜 `seam_carving.py`** → Script principal que implementa el algoritmo de reducción de costuras.
- **📜 `test_seam_carving.py`** → Funciones de test para validar el funcionamiento del algoritmo.
- **📜 `ejecutar.sh`** → Automatización de la ejecución del programa.
- **📂 `experimentacion`** → Imágenes de prueba para evaluar el algoritmo.

---

## 📌 4. Instrucciones de uso
### Conceder derechos de ejecución
```sh
chmod +x ejecutar.sh
