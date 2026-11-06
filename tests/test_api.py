"""
Suite de pruebas automatizadas para la API de FastAPI (Versión Limpia sin ruido).
Verifica endpoints de salud, metadatos, inferencia individual,
inferencia en lote, y validación de tipos Pydantic.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Cliente de pruebas que inicializa el ciclo de vida (lifespan)."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Verifica la carga de la página de inicio."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "API de Predicción de Fuga de Clientes" in response.text


def test_health_check(client):
    """Verifica que el servicio y el modelo limpio estén operativos con 9 features."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["modelo_cargado"] is True
    assert data["total_features"] == 9  # 9 variables de negocio puras
    assert "RandomForest" in data["algoritmo"]


def test_model_info(client):
    """Verifica la entrega de la ficha técnica."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "features_esperadas" in data
    assert len(data["features_esperadas"]) == 9
    assert "antiguedad_meses" in data["features_esperadas"]
    assert "ruido_1" not in data["features_esperadas"]


def test_predict_high_risk_client(client):
    """Verifica predicción de cliente crítico (sin campos de ruido)."""
    payload = {
        "id_cliente": "TEST-CRITICO-01",
        "antiguedad_meses": 2,
        "cargo_mensual": 115.0,
        "llamadas_soporte": 6,
        "uso_datos_gb": 4.5,
        "retrasos_pago": 3,
        "satisfaccion": 1.0,
        "contrato": "mensual",
        "metodo_pago": "tarjeta",
        "ciudad": "Lima"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id_cliente"] == "TEST-CRITICO-01"
    assert data["prediccion"] == 1
    assert data["probabilidad_fuga"] > 0.50
    assert data["nivel_riesgo"] in ["Alto", "Crítico"]
    assert len(data["factores_clave"]) > 0
    assert data["tiempo_inferencia_ms"] > 0


def test_predict_low_risk_client(client):
    """Verifica cliente de bajo riesgo fidelizado."""
    payload = {
        "id_cliente": "TEST-LEAL-01",
        "antiguedad_meses": 60,
        "cargo_mensual": 45.0,
        "llamadas_soporte": 0,
        "uso_datos_gb": 30.0,
        "retrasos_pago": 0,
        "satisfaccion": 9.5,
        "contrato": "bianual",
        "metodo_pago": "transferencia",
        "ciudad": "Arequipa"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediccion"] == 0
    assert data["probabilidad_fuga"] < 0.30
    assert data["nivel_riesgo"] in ["Bajo", "Moderado"]


def test_predict_validation_error_invalid_contract(client):
    """Verifica el rechazo de entradas inválidas con HTTP 422."""
    payload = {
        "antiguedad_meses": 12,
        "cargo_mensual": 50.0,
        "llamadas_soporte": 1,
        "uso_datos_gb": 10.0,
        "retrasos_pago": 0,
        "contrato": "contrato_invalido_xyz",
        "metodo_pago": "tarjeta",
        "ciudad": "Lima"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch(client):
    """Verifica inferencia en lote."""
    batch_payload = {
        "clientes": [
            {
                "id_cliente": "CLI-BATCH-1",
                "antiguedad_meses": 3,
                "cargo_mensual": 110.0,
                "llamadas_soporte": 5,
                "uso_datos_gb": 5.0,
                "retrasos_pago": 2,
                "satisfaccion": 2.0,
                "contrato": "mensual",
                "metodo_pago": "tarjeta",
                "ciudad": "Piura"
            },
            {
                "id_cliente": "CLI-BATCH-2",
                "antiguedad_meses": 50,
                "cargo_mensual": 40.0,
                "llamadas_soporte": 0,
                "uso_datos_gb": 20.0,
                "retrasos_pago": 0,
                "satisfaccion": 9.0,
                "contrato": "bianual",
                "metodo_pago": "transferencia",
                "ciudad": "Trujillo"
            }
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_evaluados"] == 2
    assert len(data["resultados"]) == 2
    assert 0.0 <= data["tasa_fuga_estimada"] <= 100.0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
