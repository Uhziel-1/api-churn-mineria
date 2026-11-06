"""
Esquemas Pydantic para la API de Predicción de Fuga de Clientes (Customer Churn).
Contiene exclusivamente variables reales del negocio (sin ruido sintético ni variables inservibles),
validaciones estrictas y ejemplos preconfigurados compatibles con Pydantic v2 y OpenAPI 3.1.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class TipoContrato(str, Enum):
    mensual = "mensual"
    anual = "anual"
    bianual = "bianual"


class TipoMetodoPago(str, Enum):
    tarjeta = "tarjeta"
    debito = "debito"
    transferencia = "transferencia"
    efectivo = "efectivo"


class TipoCiudad(str, Enum):
    Lima = "Lima"
    Piura = "Piura"
    Trujillo = "Trujillo"
    Arequipa = "Arequipa"
    Cusco = "Cusco"


class NivelRiesgo(str, Enum):
    bajo = "Bajo"
    moderado = "Moderado"
    alto = "Alto"
    critico = "Crítico"


class ClienteInput(BaseModel):
    id_cliente: Optional[str] = Field(
        default=None,
        description="Identificador opcional del cliente (referencial)",
        examples=["CLI-10293"]
    )
    antiguedad_meses: int = Field(
        ...,
        ge=1,
        le=120,
        description="Antigüedad del cliente en meses (1 a 120)",
        examples=[7]
    )
    cargo_mensual: float = Field(
        ...,
        ge=10.0,
        le=500.0,
        description="Monto facturado mensualmente al cliente en moneda local",
        examples=[85.50]
    )
    llamadas_soporte: int = Field(
        ...,
        ge=0,
        le=30,
        description="Número de llamadas realizadas al centro de atención/soporte técnico",
        examples=[4]
    )
    uso_datos_gb: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Consumo de datos móviles del último mes (en Gigabytes)",
        examples=[12.4]
    )
    retrasos_pago: int = Field(
        ...,
        ge=0,
        le=24,
        description="Cantidad de facturas con retraso en el pago en los últimos 12 meses",
        examples=[2]
    )
    satisfaccion: Optional[float] = Field(
        default=5.5,
        ge=1.0,
        le=10.0,
        description="Puntaje de satisfacción del cliente (escala 1 al 10). Si se omite, el modelo imputa automáticamente la mediana.",
        examples=[2.0]
    )
    contrato: TipoContrato = Field(
        ...,
        description="Tipo de contrato suscrito ('mensual', 'anual' o 'bianual')",
        examples=[TipoContrato.mensual]
    )
    metodo_pago: TipoMetodoPago = Field(
        ...,
        description="Método de pago habitual ('tarjeta', 'debito', 'transferencia', 'efectivo')",
        examples=[TipoMetodoPago.tarjeta]
    )
    ciudad: TipoCiudad = Field(
        ...,
        description="Ciudad donde reside el cliente",
        examples=[TipoCiudad.Lima]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id_cliente": "CLI-2026-ALTO-RIESGO",
                "antiguedad_meses": 5,
                "cargo_mensual": 98.50,
                "llamadas_soporte": 5,
                "uso_datos_gb": 8.2,
                "retrasos_pago": 2,
                "satisfaccion": 2.0,
                "contrato": "mensual",
                "metodo_pago": "tarjeta",
                "ciudad": "Lima"
            }
        }
    }


class PrediccionOutput(BaseModel):
    id_cliente: Optional[str] = Field(None, description="Identificador del cliente consultado")
    prediccion: int = Field(..., description="0 = No Fuga (Retenido), 1 = Fuga (Churn)")
    etiqueta: str = Field(..., description="Interpretación textual ('Fuga Inminente' o 'Cliente Retenido')")
    probabilidad_fuga: float = Field(..., description="Probabilidad calculada de fuga (0.0 a 1.0)")
    probabilidad_retencion: float = Field(..., description="Probabilidad calculada de retención (0.0 a 1.0)")
    nivel_riesgo: NivelRiesgo = Field(..., description="Nivel de riesgo clasificado ('Bajo', 'Moderado', 'Alto', 'Crítico')")
    factores_clave: List[str] = Field(..., description="Factores de riesgo identificados en este cliente")
    accion_recomendada: str = Field(..., description="Acción comercial y de retención recomendada")
    tiempo_inferencia_ms: float = Field(..., description="Latencia de la inferencia en milisegundos")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id_cliente": "CLI-2026-ALTO-RIESGO",
                "prediccion": 1,
                "etiqueta": "Fuga Inminente",
                "probabilidad_fuga": 0.7850,
                "probabilidad_retencion": 0.2150,
                "nivel_riesgo": "Crítico",
                "factores_clave": [
                    "Baja satisfacción reportada (2.0/10)",
                    "Alto volumen de llamadas a soporte técnico (5 incidentes)",
                    "Contrato mensual sin permanencia fija",
                    "Historial con 2 facturas con retraso"
                ],
                "accion_recomendada": "ALERTA MÁXIMA: Contactar en menos de 24h vía Customer Success. Acciones: asignar resolución técnica prioritaria nivel 2; ofrecer plan anual con 25% de descuento; bono inmediato de 20 GB de datos.",
                "tiempo_inferencia_ms": 3.42
            }
        }
    }


class BatchClienteInput(BaseModel):
    clientes: List[ClienteInput] = Field(..., description="Lista de clientes a evaluar en lote")


class BatchPrediccionOutput(BaseModel):
    total_evaluados: int = Field(..., description="Número total de clientes procesados")
    total_en_riesgo: int = Field(..., description="Clientes con predicción de fuga = 1")
    tasa_fuga_estimada: float = Field(..., description="Porcentaje de fuga detectado en el lote")
    resultados: List[PrediccionOutput] = Field(..., description="Detalle de predicción por cliente")


class HealthCheckOutput(BaseModel):
    status: str
    modelo_cargado: bool
    version_api: str
    algoritmo: str
    total_features: int


class ModelInfoOutput(BaseModel):
    nombre_modelo: str
    algoritmo: str
    clase_objetivo: str
    features_esperadas: List[str]
    categorias_validas: dict
    metricas_referencia: dict
