# Laboratorio 7. Spark MLlib
## 1. Carga, armonización y calidad de datos — 5 puntos

En esta sección cargarás los archivos Excel (convirtiéndolos a Parquet), agregarás las variables de identificación de período, aplicarás los filtros y calidad de datos requeridos, y finalmente unirás los trimestres de 2025.

### Paso 1: Configurar el entorno e importar el módulo
El código de transformación y limpieza se ha estructurado utilizando buenas prácticas en un módulo de Python ubicado en `src/data_pipeline.py`.

Abre o crea tu Notebook (Jupyter Notebook) y en la primera celda, importa PySpark y el módulo creado:

```python
import os
import sys
from pyspark.sql import SparkSession
import pandas as pd

# Agregamos la ruta para poder importar nuestro módulo personalizado
sys.path.append(os.path.abspath('./src'))
import data_pipeline as dp

# Inicializar SparkSession
spark = SparkSession.builder \
    .appName("Lab7_Spark") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
```

### Paso 2: Ejecutar el Pipeline de Carga y Filtrado
Copia el siguiente bloque de código en una celda nueva. Este script procesa los archivos uno por uno (para controlar la memoria), convirtiéndolos a Parquet, homogeneizando los tipos y calculando las variables.

```python
data_dir = "./data"
parquet_dir = "./data/parquet"
os.makedirs(parquet_dir, exist_ok=True)

# Información de cada archivo según el diccionario y sus identificadores
files_info = [
    {"file": "Personas_ENEIC_T1_2025.xlsx", "year": 2025, "quarter": 1},
    {"file": "Personas-ENEIC-T2-2025.xlsx", "year": 2025, "quarter": 2},
    {"file": "Base-de-datos-Personas-ENEIC-III-2025.xlsx", "year": 2025, "quarter": 3},
    {"file": "Base-de-datos-Personas-ENEIC-IV-2025.xlsx", "year": 2025, "quarter": 4},
    {"file": "Base-de-datos-Personas-ENEIC-I-2026.xlsx", "year": 2026, "quarter": 1}
]

dfs_processed = []
counts_info = []

for info in files_info:
    excel_path = os.path.join(data_dir, info["file"])
    parquet_path = os.path.join(parquet_dir, info["file"].replace(".xlsx", ".parquet"))
    
    # 1. Convertir de Excel a Parquet (o cargar si ya existe)
    df_raw = dp.convert_excel_to_parquet(excel_path, parquet_path, spark)
    count_before = df_raw.count()
    
    # 2. Agregar variables de período (periodo_archivo, anio_archivo, trimestre_calendario, archivo_origen)
    df_period = dp.add_period_identifiers(df_raw, info["year"], info["quarter"], info["file"])
    
    # 3. Castear columnas a los tipos requeridos y renombrarlas (salario, edad, antiguedad, etc.)
    df_clean = dp.rename_and_cast_columns(df_period)
    
    # 4. Calcular 'antiguedad' y aplicar los filtros según las instrucciones
    df_filtered = dp.calculate_and_filter(df_clean)
    count_after = df_filtered.count()
    
    # Guardar métricas de auditoría
    counts_info.append({
        "Archivo": info["file"],
        "Registros Originales": count_before,
        "Registros Filtrados": count_after
    })
    
    dfs_processed.append((info["year"], df_filtered))
    
    # Puedes usar persist si tienes memoria suficiente en el clúster
    # df_filtered.persist()
```

### Paso 3: Unir 2025 (train) y separar 2026 (test)
Pega este código para hacer un `unionByName` de los datos de entrenamiento y comprobar el total de registros.

```python
# Separar los DataFrames de 2025 (desarrollo/entrenamiento) y el de 2026 (prueba)
dfs_2025 = [df for year, df in dfs_processed if year == 2025]
df_test = [df for year, df in dfs_processed if year == 2026][0]

# Unir los 4 trimestres de 2025
df_train = dfs_2025[0]
for df in dfs_2025[1:]:
    df_train = df_train.unionByName(df)

# Cachear la tabla unida para uso posterior en el Lab
df_train.cache()

print(f"Total de registros para Entrenamiento (2025): {df_train.count()}")
print(f"Total de registros para Prueba (2026): {df_test.count()}")
```

### Paso 4: Resultados Requeridos (Muestra)
Por último, en una nueva celda, ejecuta esto para mostrar el esquema, la tabla resumen de registros eliminados por los filtros, y una muestra de los datos.

```python
# Mostrar número de registros por archivo antes y después de los filtros
counts_df = pd.DataFrame(counts_info)
display(counts_df)

# Mostrar el esquema final de los datos
df_train.printSchema()

# Mostrar 5 registros seleccionados
df_train.show(5, truncate=False)
```
