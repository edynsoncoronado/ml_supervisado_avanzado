import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from loguru import logger
from datetime import datetime
import uuid


class FeatureEngineeringProcessor:
    def __init__(self, raw_data: pd.DataFrame, pipeline_name: str) -> None:
        # Guarda el DataFrame original.
        self.raw_data = raw_data
        # Guarda el nombre del pipeline.
        self.pipeline_name = pipeline_name
        # Inicializa la tabla de características como None.
        self.feature_table = None

    def impute_scale(self, n_components: int = 2) -> pd.DataFrame:
        # Define las columnas numéricas a procesar.
        numeric_cols= [
            "lead_time",
            "adults",
            "children",
            "babies",
            "adr"
        ]
        pipe = Pipeline(
            steps=[
                # Imputa valores faltantes con la media.
                ("imputer_mean", SimpleImputer(strategy="mean")),
                # Escala las variables numéricas.
                ("std_scaling", StandardScaler()),
                # Reduce la dimensionalidad con PCA.
                ("pca", PCA(n_components=n_components))
            ]
        )
        # Devuelve un DataFrame con las nuevas características numéricas.
        return pd.DataFrame(
            pipe.fit_transform(self.raw_data[numeric_cols]),
            columns=["great_feature1", "great_feature2"]
        )

    def encode_categoricals(self) -> pd.DataFrame:
        encoded_vars = []
        for var in ["hotel", "market_segment", "reserved_room_type"]:
            # Muestra en el log qué variable se está codificando.
            logger.info(f"Codificando con OHE {var}")
            encoder = OneHotEncoder()
            # Codifica la variable categórica usando OneHotEncoder.
            encoded = encoder.fit_transform(self.raw_data[[var]]).toarray()
            cols  = [f"{var}_{col}" for col in encoder.categories_[0]]
            # Genera los nombres de las columnas codificadas.
            _dataframe = pd.DataFrame(
                encoded,
                columns= cols
            )
            # Añade el DataFrame codificado a la lista.
            encoded_vars.append(_dataframe)
        # Devuelve la concatenación de todos los DataFrames codificados.
        return pd.concat(encoded_vars,axis=1)

    def run(self) -> pd.DataFrame:
        # Log de inicio del pipeline.
        logger.info(f"Inicializando pipeline {self.pipeline_name}")

        # Codifica las variables categóricas.
        categorical = self.encode_categoricals()
        # Procesa las variables numéricas.
        numerics = self.impute_scale()

        # Une las variables categóricas y numéricas.
        modeling_dataset = pd.concat([categorical, numerics], axis=1)

        pipe = Pipeline(
            steps=[
                # Elimina variables con baja varianza.
                ("feature_selection", VarianceThreshold()),
                # Escala las variables usando RobustScaler.
                ("scaling_robust", RobustScaler())
            ]
        )
        # Aplica el pipeline y guarda el resultado en feature_table.
        self.feature_table =  pd.DataFrame(
            pipe.fit_transform(modeling_dataset),
            columns=modeling_dataset.columns
        )

        # Añade una columna de IDs únicos.
        self.feature_table["booking_id"] = [str(uuid.uuid4()) for _ in range(self.feature_table.shape[0])]
        # Añade una columna de timestamp.
        self.feature_table["event_timestamp"] = [datetime.now() for _ in range(self.feature_table.shape[0])]
        
        import time
        # Espera 1 segundo.
        time.sleep(1)
        # Añade una columna de timestamp de creación.
        self.feature_table["created"] = [datetime.now() for _ in range(self.feature_table.shape[0])]

        # Devuelve la tabla final de características.
        return self.feature_table

    def write_feature_table(self, filepath: str) -> None:
        # Log de escritura de la tabla.
        logger.info(f"Escribiendo feature table en {filepath}")
        if not self.feature_table.empty: # -> True o False
            # Guarda la tabla en formato parquet.
            self.feature_table.to_parquet(f"{filepath}.parquet", index=False)
            # Guarda la tabla en formato csv.
            self.feature_table.to_csv(f"{filepath}.csv", index=False)
        else:
            # Lanza excepción si la tabla no existe.
            raise Exception("La feature table no ha sido creada. Ejecutar el comando .run()")  