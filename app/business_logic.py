"""
Lógica de negocio y enriquecimiento de predicciones.
Transforma la probabilidad cruda de Scikit-Learn en indicadores accionables
para la toma de decisiones empresariales (CRM / Retención).
"""

from typing import List

try:
    from app.schemas import ClienteInput, NivelRiesgo
except ImportError:
    from schemas import ClienteInput, NivelRiesgo


def determinar_nivel_riesgo(probabilidad_fuga: float) -> NivelRiesgo:
    """
    Clasifica el riesgo de abandono en 4 cuadrantes estándar de CRM.
    """
    if probabilidad_fuga >= 0.70:
        return NivelRiesgo.critico
    elif probabilidad_fuga >= 0.50:
        return NivelRiesgo.alto
    elif probabilidad_fuga >= 0.25:
        return NivelRiesgo.moderado
    else:
        return NivelRiesgo.bajo


def identificar_factores_clave(datos: ClienteInput) -> List[str]:
    """
    Detecta los motivos principales de fricción basados en la importancia
    de variables del modelo RandomForest (satisfacción, soporte, contrato, antigüedad).
    """
    factores = []

    # 1. Satisfacción (Mayor predictor de fuga)
    if datos.satisfaccion is not None and datos.satisfaccion <= 3.0:
        factores.append(f"Baja satisfacción declarada por el cliente ({datos.satisfaccion:.1f}/10)")
    elif datos.satisfaccion is not None and datos.satisfaccion >= 8.0:
        factores.append(f"Alta satisfacción percibida ({datos.satisfaccion:.1f}/10)")

    # 2. Llamadas a soporte técnico
    if datos.llamadas_soporte >= 4:
        factores.append(f"Frecuencia crítica de llamadas a soporte ({datos.llamadas_soporte} incidentes reportados)")
    elif datos.llamadas_soporte >= 2:
        factores.append(f"Llamadas recurrentes a soporte ({datos.llamadas_soporte} incidentes)")

    # 3. Tipo de contrato
    if datos.contrato.value == "mensual":
        factores.append("Contrato mes a mes (baja barrera de salida / sin permanencia)")
    elif datos.contrato.value == "bianual":
        factores.append("Contrato bianual consolidado (alta fidelización)")

    # 4. Retrasos de pago
    if datos.retrasos_pago >= 2:
        factores.append(f"Historial crediticio con mora ({datos.retrasos_pago} facturas con retraso)")

    # 5. Antigüedad
    if datos.antiguedad_meses <= 6:
        factores.append(f"Cliente en fase crítica de adopción temprana (antigüedad: {datos.antiguedad_meses} meses)")
    elif datos.antiguedad_meses >= 36:
        factores.append(f"Cliente de largo recorrido ({datos.antiguedad_meses} meses en la compañía)")

    # 6. Cargo mensual
    if datos.cargo_mensual >= 100.0:
        factores.append(f"Facturación elevada ({datos.cargo_mensual:.2f} USD/mes)")

    if not factores:
        factores.append("Perfil estándar sin factores de fricción notorios")

    return factores


def sugerir_accion_comercial(
    nivel_riesgo: NivelRiesgo,
    factores: List[str],
    datos: ClienteInput
) -> str:
    """
    Formula la estrategia de retención o fidelización correspondiente.
    """
    if nivel_riesgo == NivelRiesgo.critico:
        propuestas = []
        if datos.llamadas_soporte >= 3:
            propuestas.append("asignar resolución técnica prioritaria nivel 2")
        if datos.contrato.value == "mensual":
            propuestas.append("ofrecer plan anual con 25% de descuento")
        propuestas.append("bono inmediato de 20 GB de datos")
        return f"ALERTA MÁXIMA: Contactar en menos de 24h vía Customer Success. Acciones: {'; '.join(propuestas)}."

    elif nivel_riesgo == NivelRiesgo.alto:
        if datos.contrato.value == "mensual":
            return "Riesgo Alto: Ofrecer migración a contrato anual con 15% de descuento y duplicar velocidad de datos por 3 meses."
        return "Riesgo Alto: Envío proactivo de encuesta de calidad por canal preferido y bonificación de lealtad en la siguiente factura."

    elif nivel_riesgo == NivelRiesgo.moderado:
        return "Riesgo Moderado: Seguimiento preventivo en próximo corte de facturación y verificación de estabilidad de señal/servicio."

    else:
        return "Cliente Fidelizado (Bajo Riesgo): Excelente candidato para programas VIP de lealtad, preventas exclusivas o venta cruzada (upselling)."
