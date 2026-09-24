import os
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)

def convert_excel_to_parquet(excel_path, parquet_path, spark):
    """
    Lee un archivo Excel usando pandas, lo convierte a DataFrame de Spark
    y lo guarda en formato Parquet para optimizar la lectura posterior.
    """
    if os.path.exists(parquet_path):
        print(f"El archivo parquet ya existe en: {parquet_path}")
        return spark.read.parquet(parquet_path)
        
    print(f"Leyendo {excel_path} con pandas...")
    # Leemos con pandas, convirtiendo todo a string inicialmente para evitar problemas de tipos mixtos
    pdf = pd.read_excel(excel_path, dtype=str)
    
    # Rellenar valores nulos de pandas con None para que Spark los maneje como Null
    pdf = pdf.where(pd.notnull(pdf), None)
    
    print("Convirtiendo a Spark DataFrame y guardando en Parquet...")
    df_spark = spark.createDataFrame(pdf)
    
    df_spark.write.mode("overwrite").parquet(parquet_path)
    print(f"Guardado exitosamente en: {parquet_path}")
    
    return df_spark

def add_period_identifiers(df, year, calendar_quarter, original_filename):
    """
    Agrega las columnas identificadoras de período solicitadas.
    """
    period_str = f"{year}T{calendar_quarter}"
    return df \
        .withColumn("periodo_archivo", F.lit(period_str)) \
        .withColumn("anio_archivo", F.lit(year)) \
        .withColumn("trimestre_calendario", F.lit(calendar_quarter)) \
        .withColumn("archivo_origen", F.lit(original_filename))

def rename_and_cast_columns(df):
    """
    Selecciona las columnas requeridas, las renombra y castea a los tipos correctos.
    Asegura representación consistente de los códigos antes de unir.
    """
    # Mapeo de columnas y tipos esperados
    column_mapping = {
        "P05D01": ("salario_mensual", "double"),
        "P02A03": ("edad", "integer"),
        "P05C07A": ("antiguedad_anios", "integer"),
        "P05C07B": ("antiguedad_meses", "integer"),
        "P05H01A": ("horas_semanales", "integer"),
        "P03A03A": ("nivel_educativo", "string"),
        "P05C16": ("categoria_ocupacional", "string"),
        "DOMINIO": ("dominio", "string"),
        "OCUPADOS": ("ocupado", "integer"),
        "NUM_HOGAR": ("NUM_HOGAR", "string"), 
        "NUM_PERSONA": ("NUM_PERSONA", "string"),
        "FACTOR": ("FACTOR", "double"),
        "ANIO": ("ANIO", "integer"),
        "TRIMESTRE": ("TRIMESTRE", "integer")
    }
    
    for orig_col, (new_col, cast_type) in column_mapping.items():
        if orig_col in df.columns:
            # Castear al tipo deseado y renombrar
            df = df.withColumn(new_col, F.col(orig_col).cast(cast_type))
        else:
            # Si alguna columna no viene (aunque deberían), se crea nula para evitar errores de union
            print(f"Advertencia: La columna {orig_col} no se encontró. Se creará como nula.")
            df = df.withColumn(new_col, F.lit(None).cast(cast_type))
            
    # Seleccionar solo las columnas analíticas y de auditoría (más las de periodo)
    analytic_cols = [new_col for _, (new_col, _) in column_mapping.items()]
    period_cols = ["periodo_archivo", "anio_archivo", "trimestre_calendario", "archivo_origen"]
    
    return df.select(analytic_cols + period_cols)

def calculate_and_filter(df):
    """
    Aplica cálculos y filtros solicitados en el laboratorio.
    """
    # 1. Construir antigüedad
    df = df.withColumn("antiguedad", 
                       F.col("antiguedad_anios") + (F.col("antiguedad_meses") / 12.0))
    
    # Limpieza de valores nulos o no reconocidos en categóricas -> 'DESCONOCIDO'
    # Nivel educativo 0 es 'ninguno', por lo que primero lo manejamos
    categorical_cols = ["nivel_educativo", "categoria_ocupacional", "dominio"]
    for col_name in categorical_cols:
        df = df.withColumn(col_name, F.when(F.col(col_name).isNull(), "DESCONOCIDO")
                                      .otherwise(F.col(col_name)))
    
    # 2. Filtros de población y variable objetivo
    # - De 15 años o más
    # - Identificadas como ocupadas: OCUPADOS = 1
    # - Asalariadas (categoria_ocupacional en '1','2','3','4')
    # - Salario mensual numérico, finito, estrictamente positivo
    cond_poblacion = (
        (F.col("edad") >= 15) &
        (F.col("ocupado") == 1) &
        (F.col("categoria_ocupacional").isin(["1", "2", "3", "4"])) &
        (F.col("salario_mensual") > 0)
    )
    
    # 3. Filtros de variables numéricas utilizadas (reglas lógicas y rangos)
    cond_numericas = (
        (F.col("antiguedad_anios") >= 0) &
        (F.col("antiguedad_meses") >= 0) & (F.col("antiguedad_meses") <= 11) &
        (F.col("antiguedad") <= F.col("edad")) &
        (F.col("horas_semanales") > 0) & (F.col("horas_semanales") <= 168)
    )
    
    df_filtered = df.filter(cond_poblacion & cond_numericas)
    
    return df_filtered
