# Minería de Datos - Sesión 13: Despliegue de Inferencia con FastAPI y Swagger

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Swagger](https://img.shields.io/badge/Swagger-OpenAPI_3.1-85EA2D?logo=swagger&logoColor=black)](http://127.0.0.1:8000/docs)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)

API REST de alto desempeño para la predicción en tiempo real de **Fuga de Clientes (*Customer Churn*)**. El sistema consume exclusivamente variables de negocio reales, validando las peticiones mediante Pydantic v2 y ofreciendo documentación interactiva automática en Swagger UI (`/docs`).

---

## 1. Variables de Negocio del Modelo (9 Features)

El modelo fue entrenado eliminando ruidos de laboratorio para reflejar una arquitectura de producción real:

| Variable | Tipo | Rango / Categorías | Descripción |
| :--- | :--- | :--- | :--- |
| `antiguedad_meses` | Entero | 1 a 120 meses | Tiempo de permanencia del cliente en la compañía |
| `cargo_mensual` | Decimal | 10.0 a 500.0 | Monto facturado mensualmente al cliente |
| `llamadas_soporte` | Entero | 0 a 30 llamadas | Incidentes reportados al centro de soporte |
| `uso_datos_gb` | Decimal | 0.0 a 500.0 GB | Tráfico de datos consumido en el último mes |
| `retrasos_pago` | Entero | 0 a 24 meses | Facturas con retraso en el último año |
| `satisfaccion` | Decimal | 1.0 a 10.0 | Puntaje de satisfacción (imputable si se omite) |
| `contrato` | Enum | `mensual`, `anual`, `bianual` | Modalidad de suscripción contratada |
| `metodo_pago` | Enum | `tarjeta`, `debito`, `transferencia`, `efectivo` | Medio de cobro preferente |
| `ciudad` | Enum | `Lima`, `Piura`, `Trujillo`, `Arequipa`, `Cusco` | Ubicación geográfica del titular |

---

## 2. Estructura del Proyecto

```text
07_Mineria_de_Datos/Tareas/S13/
├── app/
│   ├── __init__.py
│   ├── main.py              # Endpoints FastAPI, ciclo de vida (lifespan) y CORS
│   ├── schemas.py           # Contratos Pydantic v2 (9 variables limpias)
│   └── business_logic.py    # Lógica de scoring de riesgo y recomendaciones comerciales
├── models/
│   └── modelo_churn_limpio.joblib  # Pipeline Scikit-Learn optimizado (ROC-AUC ~ 0.803)
├── tests/
│   └── test_api.py          # Pruebas automatizadas con pytest (7 pruebas pasando)
├── Procfile                 # Descriptor de procesos para la nube
├── render.yaml              # Configuración de despliegue automatizado
├── requirements.txt         # Dependencias exactas
├── run_local.py             # Script de arranque rápido en local
├── test_client_demo.py      # Script de prueba interactiva en consola
└── README.md                # Documentación del proyecto
```

---

## 3. Ejecución Local en 1 Clic

### Paso 1: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 2: Iniciar el servidor
```bash
python run_local.py
```

### Paso 3: Abrir Swagger UI
Ingresa en tu navegador a:
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

* Despliega `POST /predict`.
* Presiona **Try it out** (los datos ya vienen precargados).
* Presiona el botón azul **Execute** y revisa la respuesta inmediata.

---

## 4. Despliegue Público con Riesgo CERO ($0.00 / Sin Tarjetas)

Para que el profesor o cualquier persona en internet pueda probar la API sin que gastes dinero ni consumas créditos:

### Opción A (Recomendada): Túnel Seguro con ngrok (Cero Nube, Control Total)
1. Con la API corriendo en local (`python run_local.py`), abre otra terminal y ejecuta:
   ```bash
   ngrok http 8000
   ```
2. ngrok te dará una URL temporal con HTTPS:  
   `https://xxxx-xx-xx.ngrok-free.app`
3. Le añades `/docs`:  
   `https://xxxx-xx-xx.ngrok-free.app/docs`
4. Se la pasas al profesor o la pones en tu entrega. En cuanto quieras cerrarla, pulsas `Ctrl + C` en tu terminal y el enlace se desactiva de inmediato. No queda ningún servidor activo ni costo asociado.

### Opción B: Hugging Face Spaces (Si requieres un link permanente)
1. Entra a [Hugging Face Spaces](https://huggingface.co/spaces) y crea un Space nuevo con SDK **Docker** o **Gradio/FastAPI**.
2. Sube los archivos de la carpeta `S13`.
3. Es 100% gratuito para siempre en el nivel CPU Basic y no solicita tarjeta de crédito.

---

## 5. Pruebas Automatizadas

Para validar que todos los endpoints y validaciones funcionen:
```bash
python -m pytest tests/test_api.py -v
```

Para ver la demostración interactiva en consola con dos clientes de prueba:
```bash
python test_client_demo.py
```

---

## 6. Ejemplo de Petición y Respuesta

### Petición `POST /predict`:
```json
{
  "id_cliente": "CLI-2026-ALTO-RIESGO",
  "antiguedad_meses": 3,
  "cargo_mensual": 115.00,
  "llamadas_soporte": 5,
  "uso_datos_gb": 6.2,
  "retrasos_pago": 3,
  "satisfaccion": 1.5,
  "contrato": "mensual",
  "metodo_pago": "tarjeta",
  "ciudad": "Lima"
}
```

### Respuesta HTTP `200 OK`:
```json
{
  "id_cliente": "CLI-2026-ALTO-RIESGO",
  "prediccion": 1,
  "etiqueta": "Fuga Inminente",
  "probabilidad_fuga": 0.7754,
  "probabilidad_retencion": 0.2246,
  "nivel_riesgo": "Crítico",
  "factores_clave": [
    "Baja satisfacción declarada por el cliente (1.5/10)",
    "Frecuencia crítica de llamadas a soporte (5 incidentes reportados)",
    "Contrato mes a mes (baja barrera de salida / sin permanencia)",
    "Historial crediticio con mora (3 facturas con retraso)",
    "Cliente en fase crítica de adopción temprana (antigüedad: 3 meses)",
    "Facturación elevada (115.00 USD/mes)"
  ],
  "accion_recomendada": "ALERTA MÁXIMA: Contactar en menos de 24h vía Customer Success. Acciones: asignar resolución técnica prioritaria nivel 2; ofrecer plan anual con 25% de descuento; bono inmediato de 20 GB de datos.",
  "tiempo_inferencia_ms": 10.21
}
```
