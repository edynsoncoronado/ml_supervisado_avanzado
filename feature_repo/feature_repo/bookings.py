from feast import FeatureStore
from feast import (
    Entity,
    FeatureView,
    FileSource,
    Field,
    RequestSource,
    FeatureService,
    PushSource,
)
from feast.types import Int64, Float64, String
from feast.on_demand_feature_view import on_demand_feature_view

import pandas as pd


# Define una entidad llamada 'booking' que se une por el campo 'booking_id'.
booking = Entity(name="booking", join_keys=["booking_id"])


# Define la fuente de datos principal desde un archivo Parquet, con campos de timestamp.
booking_source = FileSource(
    name="booking_source",
    path="../../data/processed/bookings_feature_table.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# Define una fuente de datos push que usa la fuente de datos principal.
booking_push_source = PushSource(
    name="booking_push_source",
    batch_source=booking_source,
)

# Define una fuente de datos de tipo request para recibir valores en tiempo real.
input_request = RequestSource(
    name="input_request",
    schema=[
        Field(name="kpi1", dtype=Float64),
        Field(name="kpi2", dtype=Float64),
    ],
)


# Define una vista de características con dos features, usando la entidad y la fuente de datos.
pc_booking_view = FeatureView(
    name="pc_booking_view",
    entities=[booking],
    online=True,
    schema=[
        Field(name="great_feature1", dtype=Float64, description="This is a great feature"),
        Field(name="great_feature2", dtype=Float64, description="This is a great feature"),
    ],
    source=booking_source,
)

@on_demand_feature_view(
    sources=[pc_booking_view, input_request],
    schema=[
        Field(name="great_feature1_kpi1", dtype=Float64),
        Field(name="great_feature2_kpi2", dtype=Float64),
    ],
)
def great_feature_view(inputs: pd.DataFrame) -> pd.DataFrame:
    # Crea un DataFrame vacío.
    df = pd.DataFrame()
    # Calcula una nueva feature multiplicando 'great_feature1' por 'kpi1'.
    df["great_feature1_kpi1"] = inputs["great_feature1"] * inputs["kpi1"]
    # Calcula una nueva feature multiplicando 'great_feature2' por 'kpi2'.
    df["great_feature2_kpi2"] = inputs["great_feature2"] * inputs["kpi2"]
    # Devuelve el DataFrame con las nuevas features.
    return df


# Define otro servicio de características que incluye solo la vista pc_booking_view.
fs_service_pc = FeatureService(
    name="fs_service_pc",
    features=[pc_booking_view],
)

# Define un servicio de características que incluye la fuente push y la vista on-demand.
dsrp_feature_service = FeatureService(
    name="dsrp_feature_service",
    features=[great_feature_view],
)
