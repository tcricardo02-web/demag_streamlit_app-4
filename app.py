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
    power
