"""
Servicio Web de Predicción de Fuga de Clientes (Customer Churn).
Minería de Datos - Sesión 13: Despliegue de Inferencia con FastAPI y Swagger.
Versión Limpia: Exclusivamente variables de negocio (sin datos de ruido sintético).
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys
import time
from typing import Dict, Any

# Asegurar que tanto la raíz del proyecto como el directorio app estén en sys.path
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
for d in [str(_current_dir), str(_root_dir)]:
    if d not in sys.path:
        sys.path.insert(0, d)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd

try:
    from app.schemas import (
        ClienteInput,
        PrediccionOutput,
        BatchClienteInput,
        BatchPrediccionOutput,
        HealthCheckOutput,
        ModelInfoOutput,
        TipoContrato,
        TipoMetodoPago,
        TipoCiudad,
    )
    from app.business_logic import (
        determinar_nivel_riesgo,
        identificar_factores_clave,
        sugerir_accion_comercial,
    )
except ImportError:
    from schemas import (
        ClienteInput,
        PrediccionOutput,
        BatchClienteInput,
        BatchPrediccionOutput,
        HealthCheckOutput,
        ModelInfoOutput,
        TipoContrato,
        TipoMetodoPago,
        TipoCiudad,
    )
    from business_logic import (
        determinar_nivel_riesgo,
        identificar_factores_clave,
        sugerir_accion_comercial,
    )

# Estado global del ciclo de vida
app_state: Dict[str, Any] = {
    "model": None,
    "features": [],
    "model_path": None,
    "start_time": time.time(),
}


def resolver_ruta_modelo() -> Path:
    """Busca el modelo serializado limpio en rutas relativas comunes."""
    base_dir = Path(__file__).resolve().parent
    candidatos = [
        base_dir.parent / "models" / "modelo_churn_limpio.joblib",
        base_dir / "models" / "modelo_churn_limpio.joblib",
        base_dir / "modelo_churn_limpio.joblib",
        Path.cwd() / "models" / "modelo_churn_limpio.joblib",
        Path.cwd() / "app" / "models" / "modelo_churn_limpio.joblib",
        Path.cwd() / "modelo_churn_limpio.joblib",
    ]
    for ruta in candidatos:
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        f"No se encontró 'modelo_churn_limpio.joblib'. Se buscaron: {[str(r) for r in candidatos]}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Patrón Lifespan de FastAPI: Carga el modelo una sola vez al inicializar el servidor.
    Evita sobrecarga de CPU y memoria en peticiones concurrentes.
    """
    ruta_modelo = resolver_ruta_modelo()
    print(f"[LIFESPAN] Cargando modelo Scikit-Learn limpio desde: {ruta_modelo}")
    modelo = joblib.load(ruta_modelo)
    app_state["model"] = modelo
    app_state["model_path"] = str(ruta_modelo)

    if hasattr(modelo, "feature_names_in_"):
        app_state["features"] = list(modelo.feature_names_in_)
    else:
        app_state["features"] = [
            "antiguedad_meses", "cargo_mensual", "llamadas_soporte",
            "uso_datos_gb", "retrasos_pago", "satisfaccion",
            "contrato", "metodo_pago", "ciudad"
        ]
    print(f"[LIFESPAN] Modelo cargado exitosamente. Features ({len(app_state['features'])}): {app_state['features']}")
    yield
    # Limpieza al apagar
    app_state["model"] = None
    app_state["features"] = []
    print("[LIFESPAN] Recursos del modelo liberados.")


app = FastAPI(
    title="API de Predicción de Fuga de Clientes (Customer Churn)",
    description="""
## Minería de Datos - Sesión 13: Despliegue de Inferencia con FastAPI y Swagger

Esta API proporciona inferencia en tiempo real para predecir la **probabilidad de abandono (churn)** de clientes de telecomunicaciones.

### Características del Sistema:
* **Variables 100% de Negocio (9 features):** Sin variables sintéticas ni ruido de laboratorio.
* **Pipeline Scikit-Learn:** Preprocesamiento automático (`ColumnTransformer` con imputación de mediana y `OneHotEncoder`) + `RandomForestClassifier`.
* **Enriquecimiento de Negocio:** No devuelve solo un 0 o 1; clasifica el nivel de riesgo en 4 niveles y formula acciones de retención preventivas para el equipo comercial.
* **Validación Robusta (Pydantic v2):** Validación estricta de tipos, rangos numéricos y categorías controladas (`Enum`).
* **Documentación Interactiva (Swagger/OpenAPI):** Pruébala directamente con el botón **'Try it out'** a continuación.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, tags=["Inicio"])
def index():
    """Página de bienvenida con accesos directos a Swagger y estado del servicio."""
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>API de Predicción de Churn | Minería de Datos</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 40px; display: flex; justify-content: center; align-items: center; min-height: 80vh; }
            .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 32px; max-width: 650px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); }
            h1 { color: #38bdf8; margin-top: 0; font-size: 1.8rem; }
            p { line-height: 1.6; color: #cbd5e1; }
            .badge { display: inline-block; background: #0284c7; color: #fff; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: bold; margin-bottom: 12px; }
            .btn { display: inline-block; background: #2563eb; color: white; text-decoration: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; margin-top: 16px; margin-right: 12px; transition: background 0.2s; }
            .btn:hover { background: #1d4ed8; }
            .btn-secondary { background: #475569; }
            .btn-secondary:hover { background: #334155; }
            code { background: #0f172a; padding: 2px 6px; border-radius: 4px; color: #f43f5e; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="card">
            <span class="badge">Sesión 13 - Despliegue FastAPI</span>
            <h1>API de Predicción de Fuga de Clientes</h1>
            <p>Servicio web de inferencia en producción para Minería de Datos con variables reales de negocio. Calcula el riesgo de abandono y sugiere estrategias de retención comercial.</p>
            <p><strong>Endpoints Disponibles:</strong></p>
            <ul>
                <li><code>POST /predict</code> - Inferencia individual enriquecida con análisis de riesgo.</li>
                <li><code>POST /predict/batch</code> - Inferencia masiva en lote para múltiples clientes.</li>
                <li><code>GET /health</code> - Verificación de salud y estado del modelo en memoria.</li>
                <li><code>GET /model-info</code> - Ficha técnica de variables y métricas.</li>
            </ul>
            <div style="margin-top: 24px;">
                <a href="/docs" class="btn">🚀 Abrir Swagger UI (/docs)</a>
                <a href="/redoc" class="btn btn-secondary">📖 Ver ReDoc (/redoc)</a>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthCheckOutput, tags=["Monitoreo"])
def health_check():
    """Verifica que el servicio esté activo y el modelo de machine learning esté cargado en memoria."""
    modelo_ok = app_state["model"] is not None
    algoritmo_nombre = "Desconocido"
    if modelo_ok and hasattr(app_state["model"], "named_steps"):
        step_model = app_state["model"].named_steps.get("model")
        algoritmo_nombre = type(step_model).__name__ if step_model else "Pipeline"

    return HealthCheckOutput(
        status="ok" if modelo_ok else "degraded",
        modelo_cargado=modelo_ok,
        version_api="1.0.0",
        algoritmo=algoritmo_nombre,
        total_features=len(app_state["features"])
    )


@app.get("/model-info", response_model=ModelInfoOutput, tags=["Monitoreo"])
def model_info():
    """Devuelve la ficha técnica del modelo entrenado sin variables de ruido."""
    return ModelInfoOutput(
        nombre_modelo="Optimizador de Retención Telco Churn (Variables Limpias)",
        algoritmo="RandomForestClassifier en Pipeline (Scikit-Learn)",
        clase_objetivo="churn (0 = Cliente Retenido, 1 = Fuga)",
        features_esperadas=app_state["features"],
        categorias_validas={
            "contrato": [e.value for e in TipoContrato],
            "metodo_pago": [e.value for e in TipoMetodoPago],
            "ciudad": [e.value for e in TipoCiudad],
        },
        metricas_referencia={
            "roc_auc_5_fold_cv": 0.803,
            "umbral_decision": 0.50,
            "total_estimadores": 120,
            "features_utilizadas": "9 variables puras de negocio (sin ruido sintético)"
        }
    )


def _ejecutar_inferencia_unitaria(datos: ClienteInput) -> PrediccionOutput:
    """Preprocesa y predice sobre un único registro usando exclusivamente las 9 variables de negocio."""
    modelo = app_state["model"]
    if modelo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo de machine learning no está inicializado en memoria."
        )

    t_inicio = time.perf_counter()

    # Construir DataFrame con las 9 columnas exactas que espera el Pipeline
    dict_datos = {
        "antiguedad_meses": [datos.antiguedad_meses],
        "cargo_mensual": [datos.cargo_mensual],
        "llamadas_soporte": [datos.llamadas_soporte],
        "uso_datos_gb": [datos.uso_datos_gb],
        "retrasos_pago": [datos.retrasos_pago],
        "satisfaccion": [datos.satisfaccion],
        "contrato": [datos.contrato.value],
        "metodo_pago": [datos.metodo_pago.value],
        "ciudad": [datos.ciudad.value],
    }
    df_input = pd.DataFrame(dict_datos)

    # Inferencia Scikit-Learn
    prediccion_raw = int(modelo.predict(df_input)[0])
    probabilidades = modelo.predict_proba(df_input)[0]
    p_no_fuga = float(probabilidades[0])
    p_fuga = float(probabilidades[1])

    t_fin = time.perf_counter()
    latencia_ms = round((t_fin - t_inicio) * 1000, 2)

    # Lógica de Negocio y Scoring
    nivel_riesgo = determinar_nivel_riesgo(p_fuga)
    factores = identificar_factores_clave(datos)
    accion = sugerir_accion_comercial(nivel_riesgo, factores, datos)
    etiqueta = "Fuga Inminente" if prediccion_raw == 1 else "Cliente Retenido"

    return PrediccionOutput(
        id_cliente=datos.id_cliente,
        prediccion=prediccion_raw,
        etiqueta=etiqueta,
        probabilidad_fuga=round(p_fuga, 4),
        probabilidad_retencion=round(p_no_fuga, 4),
        nivel_riesgo=nivel_riesgo,
        factores_clave=factores,
        accion_recomendada=accion,
        tiempo_inferencia_ms=latencia_ms
    )


@app.post("/predict", response_model=PrediccionOutput, tags=["Inferencia"])
def predict(datos: ClienteInput):
    """
    ### Endpoint Principal de Predicción
    Recibe las características de un cliente (9 variables de negocio) y devuelve:
    * **Predicción binaria** (0: Retenido, 1: Fuga)
    * **Probabilidad calculada** de abandono (0% a 100%)
    * **Nivel de riesgo empresarial** (Bajo, Moderado, Alto o Crítico)
    * **Factores determinantes** identificados en este cliente
    * **Acción de retención comercial recomendada**
    """
    try:
        return _ejecutar_inferencia_unitaria(datos)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el procesamiento de la inferencia: {str(exc)}"
        )


@app.post("/predict/batch", response_model=BatchPrediccionOutput, tags=["Inferencia"])
def predict_batch(payload: BatchClienteInput):
    """
    ### Endpoint de Inferencia por Lotes
    Permite evaluar un lote de clientes simultáneamente y calcula la tasa global de riesgo.
    """
    if not payload.clientes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La lista de clientes no puede estar vacía."
        )

    resultados = []
    en_riesgo = 0
    for cliente in payload.clientes:
        res = _ejecutar_inferencia_unitaria(cliente)
        if res.prediccion == 1:
            en_riesgo += 1
        resultados.append(res)

    total = len(payload.clientes)
    tasa = round((en_riesgo / total) * 100, 2)

    return BatchPrediccionOutput(
        total_evaluados=total,
        total_en_riesgo=en_riesgo,
        tasa_fuga_estimada=tasa,
        resultados=resultados
    )
