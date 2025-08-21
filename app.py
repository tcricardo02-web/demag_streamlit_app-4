import streamlit as st
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import pint

from sqlalchemy import create_engine, Column, Integer, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

# ------------------------------------------------------------------------------
# Configuração de logger e unidades (Pint)
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# ------------------------------------------------------------------------------
# Configuração do banco de dados com SQLAlchemy
# ------------------------------------------------------------------------------
DB_PATH = "sqlite:///compressor.db"
Base = declarative_base()
engine = create_engine(DB_PATH, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class FrameModel(Base):
    __tablename__ = "frame"
    id        = Column(Integer, primary_key=True, index=True)
    rpm       = Column(Float)
    stroke_m  = Column(Float)
    n_throws  = Column(Integer)
    throws    = relationship("ThrowModel", back_populates="frame")

class ThrowModel(Base):
    __tablename__ = "throw"
    id           = Column(Integer, primary_key=True, index=True)
    frame_id     = Column(Integer, ForeignKey("frame.id"))
    throw_number = Column(Integer)
    bore_m       = Column(Float)
    clearance_m  = Column(Float)
    VVCP         = Column(Float)
    SACE         = Column(Float)
    SAHE         = Column(Float)

    frame = relationship("FrameModel", back_populates="throws")

class ActuatorModel(Base):
    __tablename__ = "actuator"
    id                  = Column(Integer, primary_key=True, index=True)
    power_available_kW  = Column(Float)
    derate_percent      = Column(Float)
    air_cooler_fraction = Column(Float)

def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Banco de dados inicializado.")

# ------------------------------------------------------------------------------
# Domínio: Dataclasses para entidades
# ------------------------------------------------------------------------------
@dataclass
class Frame:
    rpm: float
    stroke: float        # metros (SI)
    n_throws: int

@dataclass
class Throw:
    throw_number: int
    bore: float          # metros
    clearance: float     # metros
    VVCP: float          # %
    SACE: float          # %
    SAHE: float          # %
    throw_id: Optional[int] = None

@dataclass
class Actuator:
    power_kW: float
    derate_percent: float
    air_cooler_fraction: float

@dataclass
class Motor:
    power_kW: float   # Potência do motor (para diagrama)

# ------------------------------------------------------------------------------
# Cálculos de performance (inspirado no Ariel 7)
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
    cp = 2.0  # kJ/(kg·K)

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
            "stage":                  stage,
            "P_in_bar":               P_in_stage / 1e5,
            "P_out_bar":              P_out_stage / 1e5,
            "PR":                     PR_base,
            "T_in_C":                 T_in - 273.15,
            "T_out_C":                T_out_actual - 273.15,
            "isentropic_efficiency":  eta_isent,
            "shaft_power_kW":         W_stage,
            "shaft_power_BHP":        W_stage * 1.34102
        })

        T_in = T_out_actual

    return {
        "mass_flow_kg_s":          m_dot,
        "inlet_pressure_bar":      P_in / 1e5,
        "inlet_temperature_C":     inlet_temperature.to(ureg.degC).magnitude,
        "n_stages":                n_stages,
        "total_shaft_power_kW":    total_W_kW,
        "total_shaft_power_BHP":   total_W_kW * 1.34102,
        "stage_details":           stage_details
    }

# ------------------------------------------------------------------------------
# Geração de diagrama interativo com Plotly
# ------------------------------------------------------------------------------
def generate_diagram(frame: Frame, throws: List[Throw], actuator: Actuator, motor: Motor) -> go.Figure:
    fig = go.Figure()
    width, height = 900, 350

    # Motor (esquerda)
    m_x, m_y, m_w, m_h = 30, height/2-25, 100, 50
    fig.add_shape("rect", x0=m_x, y0=m_y, x1=m_x+m_w, y1=m_y+m_h,
                  line=dict(color="MediumPurple"), fillcolor="Lavender")
    fig.add_annotation(x=m_x+m_w/2, y=m_y+m_h/2,
                       text=f"Motor<br>{motor.power_kW*1.34102:.0f} BHP",
                       showarrow=False, font=dict(size=12), align="center")

    # Frame (centro)
    f_x, f_y, f_w, f_h = m_x+m_w+50, height/2-25, 200, 50
    fig.add_shape("rect", x0=f_x, y0=f_y, x1=f_x+f_w, y1=f_y+f_h,
                  line=dict(color="RoyalBlue"), fillcolor="LightSkyBlue")
    fig.add_annotation(x=f_x+f_w/2, y=f_y+f_h/2,
                       text=f"Frame<br>RPM: {frame.rpm:.0f}",
                       showarrow=False, font=dict(size=12), align="center")

    # Throws (embaixo do frame)
    n = len(throws)
    spacing = f_w / n if n else 0
    for t in throws:
        idx = t.throw_number - 1
        tx = f_x + idx*spacing + spacing/4
        ty = f_y + f_h + 20
        tw, th = spacing/2, 30
        fig.add_shape("rect", x0=tx, y0=ty, x1=tx+tw, y1=ty+th,
                      line=dict(color="DarkOrange"), fillcolor="Moccasin")
        fig.add_annotation(x=tx+tw/2, y=ty+th/2,
                           text=f"Throw {t.throw_number}",
                           showarrow=False, font=dict(size=10))

    # Atuador (direita)
    a_x, a_y, a_w, a_h = f_x+f_w+50, height/2-20, 120, 60
    fig.add_shape("rect", x0=a_x, y0=a_y, x1=a_x+a_w, y1=a_y+a_h,
                  line=dict(color="SaddleBrown"), fillcolor="PeachPuff")
    fig.add_annotation(x=a_x+a_w/2, y=a_y+a_h/2,
                       text=f"Acionador<br>{actuator.power_kW:.0f} kW",
                       showarrow=False, font=dict(size=12), align="center")

    fig.update_layout(width=width, height=height,
                      margin=dict(l=20, r=20, t=20, b=20),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# ------------------------------------------------------------------------------
# Interface do usuário com Streamlit
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Calculadora de Performance de Compressor", layout="wide")
    st.title("Calculadora de Performance de Compressor (Estilo Ariel 7)")

    init_db()

    UNIT_OPTIONS = ["SI", "Metric"]
    if "unit_system" not in st.session_state:
        st.session_state["unit_system"] = "SI"

    with st.sidebar:
        st.header("Configurações Gerais")
        unit = st.selectbox("Sistema de unidades", UNIT_OPTIONS, index=0)
        st.session_state["unit_system
