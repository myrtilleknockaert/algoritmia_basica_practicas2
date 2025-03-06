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
### 📂 `practica2_950123_950134/`
El directorio contiene:
- **📜 `README.md`** → Explicación del proyecto (este archivo).
- **📜 `seam_carving.py`** → Script principal que implementa el algoritmo de reducción de costuras.
- **📜 `ejecutar.sh`** → Automatización de la ejecución del programa.
- **📂 `experimentacion`** → Imágenes de prueba para evaluar el algoritmo.

---

## 📌 4. Instrucciones de uso

### Grant Execution Rights and Prepare the Environment

First, make the `ejecutar.sh` script executable and run it to set up the environment:

```sh
chmod +x ejecutar.sh
./ejecutar.sh
```

## 📌 5. Ejecutar las pruebas

Para probar la ejecución del algoritmo de reducción de costuras con una imagen de prueba, asegúrese de haber preparado el entorno siguiendo las instrucciones de la sección 4. Luego, ejecute el siguiente comando desde el directorio raíz del proyecto:

```sh
./seam_carving.py 50 experimentacion/elefante.jpg ./
