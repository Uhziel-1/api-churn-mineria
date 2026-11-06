"""
Script para iniciar la API localmente en modo desarrollo con recarga automática.
Ejecución:
    python run_local.py
"""

from pathlib import Path
import sys
import uvicorn

CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 INICIANDO SERVIDOR FASTAPI - MINERÍA DE DATOS S13")
    print("=" * 65)
    print(f"📁 Directorio base: {CURRENT_DIR}")
    print(f"📖 Documentación interactiva Swagger: http://127.0.0.1:8000/docs")
    print(f"📖 Documentación ReDoc:              http://127.0.0.1:8000/redoc")
    print(f"🩺 Estado de Salud (Health):          http://127.0.0.1:8000/health")
    print("=" * 65)
    print("Presiona Ctrl + C para detener el servidor.\n")

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=str(APP_DIR)
    )
