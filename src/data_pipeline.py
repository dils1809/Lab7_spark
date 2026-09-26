import os
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType

# Columnas de la base de Personas que se utilizan (nombre original)
COLS_FUENTE = ["ANIO", "TRIMESTRE", "DOMINIO", "NUM_HOGAR", "NUM_PERSONA", "FACTOR", "OCUPADOS",
               "P02A03", "P03A03A", "P05C07A", "P05C07B", "P05C16", "P05D01", "P05H01A"]

# periodo_archivo, anio_archivo, trimestre_calendario, archivo, uso
ARCHIVOS = [
    ("2025T1", 2025, 1, "Personas_ENEIC_T1_2025.xlsx",                "entrenamiento"),
    ("2025T2", 2025, 2, "Personas-ENEIC-T2-2025.xlsx",                "entrenamiento"),
    ("2025T3", 2025, 3, "Base-de-datos-Personas-ENEIC-III-2025.xlsx", "entrenamiento"),
    ("2025T4", 2025, 4, "Base-de-datos-Personas-ENEIC-IV-2025.xlsx",  "entrenamiento"),
    ("2026T1", 2026, 1, "Base-de-datos-Personas-ENEIC-I-2026.xlsx",   "prueba"),
]
REGISTROS_ESPERADOS = {"2025T1": 51588, "2025T2": 51167, "2025T3": 51583, "2025T4": 49338, "2026T1": 49843}

# Códigos según los diccionarios de datos (idénticos en los cinco trimestres)
NIVEL_EDUCATIVO = {0: "Ninguno", 1: "Preprimaria", 2: "Primaria", 3: "Básico", 4: "Diversificado",
                   5: "Superior", 6: "Maestría", 7: "Doctorado"}
CATEGORIA_OCUPACIONAL = {1: "Empleado de gobierno", 2: "Empleado empresa privada",
                         3: "Jornalero o peón", 4: "Servicio doméstico"}
DOMINIO = {1: "Urbano Metropolitano", 2: "Resto Urbano", 3: "Rural Nacional"}
DESCONOCIDO = "DESCONOCIDO"

TEXTOS_NULOS = ["", "nan", "NaN", "NAN", "None", "NONE", "null", "NULL", "NA", "N/A", "."]


def clean_text(c):
    """Texto recortado; vacío/'nan'/'None' se tratan como nulo."""
    s = F.trim(F.col(c))
    return F.when(s.isNull() | s.isin(TEXTOS_NULOS), None).otherwise(s)


def to_number(c):
    return clean_text(c).cast("double")


def to_code(c):
    """Código entero: '2', 2 y '2.0' producen 2; cualquier otra cosa produce nulo."""
    d = to_number(c)
    return F.when(d.isNotNull() & ~F.isnan(d) & (F.abs(d) < 1e15) & (d == F.floor(d)), d.cast("long"))


def to_identifier(c):
    """Clave estable: '21052', 21052 y '21052.0' producen la misma cadena."""
    return F.coalesce(to_code(c).cast("string"), clean_text(c))


def read_excel_columns(excel_path, parquet_path, spark, usecols=COLS_FUENTE):
    """
    Igual que convert_excel_to_parquet, pero lee solo las columnas requeridas (menos memoria)
    y usa un esquema explícito de texto, para no depender de la inferencia de tipos de Spark.
    """
    if os.path.exists(parquet_path):
        return spark.read.parquet(parquet_path)
    pdf = pd.read_excel(excel_path, dtype=str, usecols=lambda c: str(c).strip() in usecols)
    pdf.columns = [str(c).strip() for c in pdf.columns]
    pdf = pdf[list(usecols)].astype(object).where(pdf.notna(), None)
    schema = StructType([StructField(c, StringType(), True) for c in usecols])
    spark.createDataFrame(pdf, schema=schema).write.mode("overwrite").parquet(parquet_path)
    del pdf
    return spark.read.parquet(parquet_path)


def missing_report(df_raw):
    """Nulos y valores no convertibles por columna original, sobre el archivo completo (antes de filtros)."""
    aggs = [F.count(F.lit(1)).alias("n")]
    for c in COLS_FUENTE:
        aggs.append(F.sum(clean_text(c).isNull().cast("int")).alias(f"{c}__nulo"))
        aggs.append(F.sum((clean_text(c).isNotNull() & to_number(c).isNull()).cast("int")).alias(f"{c}__no_convertible"))
    fila = df_raw.agg(*aggs).first().asDict()
    return [{"variable_original": c, "n": fila["n"], "nulos": fila[f"{c}__nulo"],
             "no_convertibles": fila[f"{c}__no_convertible"]} for c in COLS_FUENTE]


def harmonize(df_raw, year, quarter, filename):
    """
    Tipifica de forma explícita las columnas requeridas y agrega la identificación del período
    según el archivo de procedencia (TRIMESTRE se conserva tal cual, no se usa como trimestre calendario).
    """
    return df_raw.select(
        F.lit(filename).alias("archivo_origen"),
        F.lit(f"{year}T{quarter}").alias("periodo_archivo"),
        F.lit(year).cast("int").alias("anio_archivo"),
        F.lit(quarter).cast("int").alias("trimestre_calendario"),
        to_code("ANIO").cast("int").alias("ANIO"),
        to_code("TRIMESTRE").cast("int").alias("TRIMESTRE"),
        to_identifier("NUM_HOGAR").alias("NUM_HOGAR"),
        to_identifier("NUM_PERSONA").alias("NUM_PERSONA"),
        to_number("FACTOR").alias("FACTOR"),
        to_code("DOMINIO").cast("int").alias("dominio_cod"),
        to_code("OCUPADOS").cast("int").alias("ocupado"),
        to_number("P02A03").alias("edad"),
        to_code("P03A03A").cast("int").alias("nivel_educativo_cod"),
        to_number("P05C07A").alias("antiguedad_anios"),
        to_number("P05C07B").alias("antiguedad_meses"),
        to_code("P05C16").cast("int").alias("categoria_ocupacional_cod"),
        to_number("P05D01").alias("salario_mensual"),
        to_number("P05H01A").alias("horas_semanales"),
    )


def _finite(c):
    return F.col(c).isNotNull() & ~F.isnan(F.col(c)) & (F.abs(F.col(c)) != float("inf"))


# Se aplican siempre en este orden; un registro se excluye en el PRIMER paso que no cumple.
# Las condiciones se construyen al llamar (no al importar) porque F.col requiere una SparkSession activa.
PASOS_NOMBRES = ["1. Edad finita y >= 15", "2. Ocupado (OCUPADOS = 1)", "3. Asalariado (P05C16 en 1-4)",
                 "4. Salario finito y > 0", "5. Años de antigüedad >= 0", "6. Meses enteros entre 0 y 11",
                 "7. Antigüedad <= edad", "8. Horas > 0 y <= 168"]


def filter_steps():
    """Lista ordenada de (nombre, condición) de la población analítica."""
    condiciones = [
        _finite("edad") & (F.col("edad") >= 15),
        F.col("ocupado") == 1,
        F.col("categoria_ocupacional_cod").isin(1, 2, 3, 4),
        _finite("salario_mensual") & (F.col("salario_mensual") > 0),
        _finite("antiguedad_anios") & (F.col("antiguedad_anios") >= 0),
        _finite("antiguedad_meses") & (F.col("antiguedad_meses") == F.floor("antiguedad_meses"))
        & F.col("antiguedad_meses").between(0, 11),
        F.col("antiguedad") <= F.col("edad"),
        _finite("horas_semanales") & (F.col("horas_semanales") > 0) & (F.col("horas_semanales") <= 168),
    ]
    return list(zip(PASOS_NOMBRES, condiciones))


def mark_exclusions(df):
    """Agrega 'antiguedad' (años) y 'motivo_exclusion' (nulo si el registro cumple todos los pasos)."""
    df = df.withColumn("antiguedad", F.col("antiguedad_anios") + F.col("antiguedad_meses") / 12.0)
    motivo = F.lit(None).cast("string")
    for nombre, condicion in reversed(filter_steps()):
        motivo = F.when(~F.coalesce(condicion, F.lit(False)), F.lit(nombre)).otherwise(motivo)
    return df.withColumn("motivo_exclusion", motivo)


def _label(codigo_col, diccionario):
    mapa = F.create_map(*[x for k, v in diccionario.items() for x in (F.lit(k), F.lit(f"{k} - {v}"))])
    return F.coalesce(mapa[F.col(codigo_col)], F.lit(DESCONOCIDO))


COLS_FINALES = ["archivo_origen", "periodo_archivo", "anio_archivo", "trimestre_calendario", "ANIO", "TRIMESTRE",
                "NUM_HOGAR", "NUM_PERSONA", "FACTOR", "ocupado",
                "edad", "antiguedad_anios", "antiguedad_meses", "antiguedad", "horas_semanales", "salario_mensual",
                "nivel_educativo", "categoria_ocupacional", "dominio",
                "nivel_educativo_cod", "categoria_ocupacional_cod", "dominio_cod"]


def prepare_dataset(df_marked):
    """Conserva los registros que cumplen todos los pasos y valida las categóricas contra el diccionario."""
    return (df_marked.filter(F.col("motivo_exclusion").isNull())
            .withColumn("nivel_educativo", _label("nivel_educativo_cod", NIVEL_EDUCATIVO))
            .withColumn("categoria_ocupacional", _label("categoria_ocupacional_cod", CATEGORIA_OCUPACIONAL))
            .withColumn("dominio", _label("dominio_cod", DOMINIO))
            .select(COLS_FINALES))
