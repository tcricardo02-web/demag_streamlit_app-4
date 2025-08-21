import streamlit as st
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import plotly.graph_objects as go
import pint
import io
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

import tempfile

# ------------------------------------------------------------------------------
# Logger e Unidades
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# ------------------------------------------------------------------------------
# Banco de dados
# ------------------------------------------------------------------------------
DB_PATH = "sqlite:///compressor.db"
Base = declarative_base()
engine = create_engine(DB_PATH, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class FrameModel(Base):
    __tablename__ = "frame"
    id = Column(Integer, primary_key=True, index=True)
    rpm = Column(Float)
    stroke_m = Column(Float)
    n_throws = Column(Integer)
    throws = relationship("ThrowModel", back_populates="frame")

class ThrowModel(Base):
    __tablename__ = "throw"
    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(Integer, ForeignKey("frame.id"))
    throw_number = Column(Integer)
    bore_m = Column(Float)
    clearance_m = Column(Float)
    VVCP = Column(Float)
    SACE = Column(Float)
    SAHE = Column(Float)

    frame = relationship("FrameModel", back_populates="throws")

class ActuatorModel(Base):
    __tablename__ = "actuator"
    id = Column(Integer, primary_key=True, index=True)
    power_available_kW = Column(Float)
    derate_percent = Column(Float)
    air_cooler_fraction = Column(Float)

def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Banco de dados inicializado.")

# ------------------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------------------
@dataclass
class Frame:
    rpm: float
    stroke: float
    n_throws: int

@dataclass
class Throw:
    throw_number: int
    bore: float
    clearance: float
    VVCP: float
    SACE: float
    SAHE: float
    throw_id: Optional[int] = None

@dataclass
class Actuator:
    power_kW: float
    derate_percent: float
    air_cooler_fraction: float

@dataclass
class Motor:
    power_kW: float

# ------------------------------------------------------------------------------
# Cálculos de performance
# ------------------------------------------------------------------------------
def clamp(n, a, b):
    return max(a, min(n, b))

def perform_performance_calculation(
    mass_flow: float,
    inlet_pressure: Q_,
    inlet_temperature: Q_,
    n_stages: int,
    PR_total: float,
    throws: List[Throw],
    stage_mapping: Dict[int, List[int]],
    actuator: Actuator,
) -> Dict:
    m_dot = mass_flow
    P_in = inlet_pressure.to(ureg.Pa).magnitude
    T_in = inlet_temperature.to(ureg.K).magnitude

    n = max(n_stages, 1)
    PR_base = PR_total ** (1.0 / n)

    gamma = 1.30
    cp = 2.0

    stage_details = []
    total_W_kW = 0.0
    throws_by_number = {t.throw_number: t for t in throws}

    for stage in range(1, n + 1):
        P_in_stage = P_in * (PR_base ** (stage - 1))
        P_out_stage = P_in_stage * PR_base

        assigned = stage_mapping.get(stage, [])
        if assigned:
            SACE_avg = sum(throws_by_number[t].SACE for t in assigned if t in throws_by_number) / len(assigned)
            VVCP_avg = sum(throws_by_number[t].VVCP for t in assigned if t in throws_by_number) / len(assigned)
            SAHE_avg = sum(throws_by_number[t].SAHE for t in assigned if t in throws_by_number) / len(assigned)
        else:
            SACE_avg = VVCP_avg = SAHE_avg = 0.0

        eta_isent = 0.65 + 0.15 * (SACE_avg / 100.0) - 0.05 * (VVCP_avg / 100.0) + 0.10 * (SAHE_avg / 100.0)
        eta_isent = clamp(eta_isent, 0.65, 0.92)

        T_out_isent = T_in * (PR_base ** ((gamma - 1.0) / gamma))
        T_out_actual = T_in + (T_out_isent - T_in) / max(eta_isent, 1e-6)
        delta_T = T_out_actual - T_in

        W_stage = m_dot * cp * delta_T / 1000.0
        total_W_kW += W_stage

        stage_details.append({
            "stage": stage,
            "P_in_bar": P_in_stage / 1e5,
            "P_out_bar": P_out_stage / 1e5,
            "PR": PR_base,
            "T_in_C": T_in - 273.15,
            "T_out_C": T_out_actual - 273.15,
            "isentropic_efficiency": eta_isent,
            "shaft_power_kW": W_stage,
            "shaft_power_BHP": W_stage * 1.34102
        })

        T_in = T_out_actual

    return {
        "mass_flow_kg_s": m_dot,
        "inlet_pressure_bar": P_in / 1e5,
        "inlet_temperature_C": inlet_temperature.to(ureg.degC).magnitude,
        "n_stages": n_stages,
        "total_shaft_power_kW": total_W_kW,
        "total_shaft_power_BHP": total_W_kW * 1.34102,
        "stage_details": stage_details
    }

# ------------------------------------------------------------------------------
# Diagrama com Plotly
# ------------------------------------------------------------------------------
def generate_diagram(frame: Frame, throws: List[Throw], actuator: Actuator, motor: Motor) -> go.Figure:
    fig = go.Figure()
    width, height = 900, 350

    fig.add_shape(type="rect", x0=30, y0=height/2-25, x1=130, y1=height/2+25,
                  line=dict(color="MediumPurple"), fillcolor="Lavender")
    fig.add_annotation(x=80, y=height/2,
                       text=f"Motor<br>{motor.power_kW*1.34102:.0f} BHP",
                       showarrow=False, font=dict(size=12), align="center")

    f_x, f_y, f_w, f_h = 180, height/2-25, 200, 50
    fig.add_shape(type="rect", x0=f_x, y0=f_y, x1=f_x+f_w, y1=f_y+f_h,
                  line=dict(color="RoyalBlue"), fillcolor="LightSkyBlue")
    fig.add_annotation(x=f_x+f_w/2, y=f_y+f_h/2,
                       text=f"Frame<br>RPM: {frame.rpm:.0f}",
                       showarrow=False, font=dict(size=12), align="center")

    n = len(throws)
    spacing = f_w / n if n else 0
    for t in throws:
        idx = t.throw_number - 1
        tx = f_x + idx*spacing + spacing/4
        ty = f_y + f_h + 20
        tw, th = spacing/2, 30
        fig.add_shape(type="rect", x0=tx, y0=ty, x1=tx+tw, y1=ty+th,
                      line=dict(color="DarkOrange"), fillcolor="Moccasin")
        fig.add_annotation(x=tx+tw/2, y=ty+th/2,
                           text=f"Throw {t.throw_number}",
                           showarrow=False, font=dict(size=10))

    a_x, a_y, a_w, a_h = f_x+f_w+50, height/2-20, 120, 60
    fig.add_shape(type="rect", x0=a_x, y0=a_y, x1=a_x+a_w, y1=a_y+a_h,
                  line=dict(color="SaddleBrown"), fillcolor="PeachPuff")
    fig.add_annotation(x=a_x+a_w/2, y=a_y+a_h/2,
                       text=f"Acionador<br>{actuator.power_kW:.0f} kW",
                       showarrow=False, font=dict(size=12), align="center")

    fig.update_layout(width=width, height=height,
                      margin=dict(l=20, r=20, t=20, b=20),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# ------------------------------------------------------------------------------
# PDF Export
# ------------------------------------------------------------------------------
def export_to_pdf(results: Dict, fig: go.Figure) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    normal = styles["Normal"]

    # Cabeçalho com placeholder
    elements.append(Paragraph("<b>[LOGO AQUI]</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Relatório de Performance do Compressor", title_style))
    elements.append(Spacer(1, 12))

    # Resumo
    summary = [
        ["Massa (kg/s)", f"{results['mass_flow_kg_s']:.2f}"],
        ["Pressão In (bar)", f"{results['inlet_pressure_bar']:.2f}"],
        ["Temp In (°C)", f"{results['inlet_temperature_C']:.2f}"],
        ["Estágios", f"{results['n_stages']}"],
        ["Potência Total (kW)", f"{results['total_shaft_power_kW']:.2f}"],
        ["Potência Total (BHP)", f"{results['total_shaft_power_BHP']:.2f}"]
    ]
    table = Table(summary)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Diagrama como imagem
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.write_image(tmpfile.name, format="png")
        elements.append(Image(tmpfile.name, width=400, height=160))
        elements.append(Spacer(1, 12))

    # Detalhes de estágio
    data = [["Estágio", "P_in (bar)", "P_out (bar)", "PR", "T_in (°C)", "T_out (°C)", "Eficiência", "Potência (kW)"]]
    for s in results["stage_details"]:
        data.append([
            s["stage"], f"{s['P_in_bar']:.2f}", f"{s['P_out_bar']:.2f}", f"{s['PR']:.2f}",
            f"{s['T_in_C']:.1f}", f"{s['T_out_C']:.1f}", f"{s['isentropic_efficiency']:.2f}", f"{s['shaft_power_kW']:.2f}"
        ])
    stage_table = Table(data)
    stage_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    elements.append(stage_table)
    elements.append(Spacer(1, 24))

    # Rodapé
    footer = f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} por CompressorCalc"
    elements.append(Paragraph(footer, normal))

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# ------------------------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Calculadora de Performance de Compressor", layout="wide")
    st.title("Calculadora de Performance de Compressor (Estilo Ariel 7)")

    init_db()

    with st.sidebar:
        st.header("Configurações Gerais")
        if st.button("Resetar DB"):
            import os
            if os.path.exists("compressor.db"):
                os.remove("compressor.db")
            init_db()
            st.success("Banco de dados reinicializado.")

    st.subheader("Definições do Compressor")
    rpm = st.number_input("RPM", value=900)
    stroke = st.number_input("Curso (m)", value=0.2)
    n_throws = st.number_input("Número de Throws", value=2, min_value=1, step=1)

    throws_data = []
    for i in range(1, n_throws+1):
        throws_data.append({"throw_number": i, "bore": 0.2, "clearance": 0.02, "VVCP": 5.0, "SACE": 5.0, "SAHE": 5.0})
    df_throws = pd.DataFrame(throws_data)
    df_throws = st.data_editor(df_throws, num_rows="dynamic")

    actuator_power = st.number_input("Potência disponível (kW)", value=500.0)
    actuator_derate = st.number_input("Derate (%)", value=0.0)
    actuator_aircooler = st.number_input("Fração Air Cooler", value=0.1)

    mass_flow = st.number_input("Massa de entrada (kg/s)", value=10.0)
    inlet_pressure = st.number_input("Pressão de entrada (bar)", value=1.0)
    inlet_temperature = st.number_input("Temperatura de entrada (°C)", value=25.0)
    n_stages = st.number_input("Número de estágios", value=2, min_value=1, step=1)
    PR_total = st.number_input("Relação de pressão total", value=4.0)

    frame = Frame(rpm, stroke, n_throws)
    throws = [Throw(int(r.throw_number), float(r.bore), float(r.clearance), float(r.VVCP), float(r.SACE), float(r.SAHE)) for r in df_throws.itertuples(index=False)]
    actuator = Actuator(actuator_power, actuator_derate, actuator_aircooler)
    motor = Motor(actuator_power)

    stage_mapping = {i: [i] for i in range(1, n_stages+1)}

    if st.button("Calcular Performance"):
        results = perform_performance_calculation(
            mass_flow,
            Q_(inlet_pressure, ureg.bar),
            Q_(inlet_temperature, ureg.degC),
            n_stages,
            PR_total,
            throws,
            stage_mapping,
            actuator,
        )

        st.subheader("Resultados")
        st.json(results)

        fig = generate_diagram(frame, throws, actuator, motor)
        st.plotly_chart(fig, use_container_width=True)

        df_results = pd.DataFrame(results["stage_details"])
        st.dataframe(df_results)

        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Baixar CSV", csv, "resultados.csv", "text/csv")

        xlsx_buffer = io.BytesIO()
        with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
            df_results.to_excel(writer, index=False)
        st.download_button("⬇️ Baixar Excel", xlsx_buffer.getvalue(), "resultados.xlsx")

        pdf_bytes = export_to_pdf(results, fig)
        st.download_button
