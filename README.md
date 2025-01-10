# Clasificación de Imágenes de Perros y Gatos mediante Redes Neuronales Convolucionales

Este proyecto implementa un sistema avanzado de clasificación de imágenes utilizando Redes Neuronales Convolucionales (CNN) para distinguir automáticamente entre fotografías de perros y gatos. La solución desarrollada emplea una arquitectura CNN moderna con TensorFlow y Keras, incorporando técnicas de deep learning estado del arte para maximizar la precisión de la clasificación.

La arquitectura del modelo se basa en una CNN profunda que incluye cuatro bloques convolucionales, cada uno compuesto por capas Conv2D, BatchNormalization y MaxPooling2D. Para mejorar la generalización y prevenir el sobreajuste, se implementaron múltiples técnicas de regularización, incluyendo Dropout al 50%, regularización L2 con un lambda de 0.001, y técnicas de data augmentation que incluyen rotaciones, zoom y volteos horizontales de las imágenes.

El sistema fue entrenado con un dataset de 30,000 imágenes de alta calidad, redimensionadas a 150x150 píxeles para mantener un equilibrio entre la calidad de la información y la eficiencia computacional. El conjunto de datos se dividió estratégicamente en tres subconjuntos: 70% para entrenamiento, 15% para validación durante el entrenamiento, y 15% para pruebas finales. El modelo alcanzó una precisión del 89.29% en el conjunto de prueba, demostrando su robustez y capacidad de generalización.

La implementación incluye características avanzadas como early stopping para prevenir el sobreajuste, reducción adaptativa de la tasa de aprendizaje mediante ReduceLROnPlateau, y un sistema de checkpoints para guardar los mejores modelos durante el entrenamiento. Estas características, combinadas con la arquitectura optimizada del modelo, permiten una clasificación precisa y eficiente de nuevas imágenes de perros y gatos.

¿Te gustaría que expandiera algún aspecto específico de la descripción?

## Descripción
El modelo utiliza una arquitectura CNN con múltiples capas convolucionales, normalización por lotes y regularización L2 para prevenir el sobreajuste. El proyecto alcanza una precisión del 89.29% en el conjunto de prueba.

## Estructura del Modelo
- 4 bloques convolucionales con:
  - Capas Conv2D
  - Batch Normalization
  - Activación ReLU
  - MaxPooling2D
- Capa Flatten
- Dropout (0.5)
- Capa Dense final con activación sigmoid

## Dataset
- Dataset de 30,000 imágenes de perros y gatos (150x150 píxeles)
- Split de datos:
  - Entrenamiento: 70%
  - Validación: 15%
  - Prueba: 15%

## Técnicas de Optimización
- Data Augmentation:
  - Rotación
  - Desplazamiento horizontal
  - Zoom
  - Volteo horizontal
- Regularización L2 (lambda=0.001)
- Batch Normalization
- Dropout (50%)
- Early Stopping
- Learning Rate Reduction
- Checkpoints para guardar el mejor modelo

## Requisitos
```python
tensorflow
keras
numpy
pandas
matplotlib
kagglehub
```

## Estructura del Proyecto
```
.
├── model_cat.keras         # Modelo entrenado
├── best_dogs_cats_model.keras   # Mejor modelo guardado
├── train/
│   ├── cats/
│   └── dogs/
├── validation/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

## Resultados
- Precisión en entrenamiento: 89.25%
- Precisión en prueba: 89.29%

## Uso
1. Clonar el repositorio
2. Instalar dependencias:
```bash
pip install -r requirements.txt
```
3. Ejecutar el notebook o script principal
4. Para predicciones:
```python
from keras.models import load_model
model = load_model('best_dogs_cats_model.keras')
# Preparar imagen y hacer predicción
```

## Mejoras Futuras
- Implementar transfer learning con modelos pre-entrenados
- Aumentar el tamaño del dataset
- Probar diferentes arquitecturas de CNN
- Implementar interfaz de usuario para predicciones

## Autor
Eduard Manuel Giraldo Martinez
