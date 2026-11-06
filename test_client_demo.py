"""
Cliente de demostración interactivo para la API de Churn (Versión Limpia).
Envía perfiles reales de clientes a la API y muestra la respuesta en consola.
Uso:
    python test_client_demo.py
"""

from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from fastapi.testclient import TestClient
from main import app

CASOS_DE_PRUEBA = [
    {
        "titulo": "CASO 1: Cliente en Riesgo Crítico de Fuga",
        "datos": {
            "id_cliente": "CLI-CRITICO-99",
            "antiguedad_meses": 3,
            "cargo_mensual": 115.0,
            "llamadas_soporte": 5,
            "uso_datos_gb": 6.2,
            "retrasos_pago": 3,
            "satisfaccion": 1.5,
            "contrato": "mensual",
            "metodo_pago": "tarjeta",
            "ciudad": "Lima"
        }
    },
    {
        "titulo": "CASO 2: Cliente Leal y Fidelizado",
        "datos": {
            "id_cliente": "CLI-FIDELIZADO-01",
            "antiguedad_meses": 58,
            "cargo_mensual": 48.0,
            "llamadas_soporte": 0,
            "uso_datos_gb": 35.0,
            "retrasos_pago": 0,
            "satisfaccion": 9.5,
            "contrato": "bianual",
            "metodo_pago": "transferencia",
            "ciudad": "Arequipa"
        }
    }
]

def main():
    print("=" * 70)
    print("DEMOSTRACIÓN DE INFERENCIA EN FASTAPI (MODELO LIMPIO - 9 VARIABLES)")
    print("=" * 70)

    with TestClient(app) as client:
        for caso in CASOS_DE_PRUEBA:
            print(f"\n>>> {caso['titulo']}")
            print(f"ID Cliente: {caso['datos']['id_cliente']}")
            print(f"Parámetros: Antigüedad {caso['datos']['antiguedad_meses']}m | Factura: S/.{caso['datos']['cargo_mensual']} | Soporte: {caso['datos']['llamadas_soporte']} llamadas | Contrato: {caso['datos']['contrato']}")

            resp = client.post("/predict", json=caso["datos"])
            if resp.status_code == 200:
                resultado = resp.json()
                print("\n--- RESPUESTA DEL ENDPOINT ---")
                print(f"Predicción Binaria:       {resultado['prediccion']} ({resultado['etiqueta']})")
                print(f"Probabilidad de Fuga:     {resultado['probabilidad_fuga'] * 100:.2f}%")
                print(f"Probabilidad Retención:   {resultado['probabilidad_retencion'] * 100:.2f}%")
                print(f"Nivel de Riesgo:          [{resultado['nivel_riesgo']}]")
                print(f"Latencia de Inferencia:   {resultado['tiempo_inferencia_ms']} ms")
                print("Factores Críticos Detectados:")
                for factor in resultado['factores_clave']:
                    print(f"   • {factor}")
                print(f"Acción Comercial Sugerida:\n   -> {resultado['accion_recomendada']}")
            else:
                print(f"Error {resp.status_code}: {resp.text}")

    print("\n" + "=" * 70)
    print("DEMOSTRACIÓN FINALIZADA CON ÉXITO")
    print("=" * 70)

if __name__ == "__main__":
    main()
