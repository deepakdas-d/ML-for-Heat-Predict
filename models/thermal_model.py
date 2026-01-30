from dataclasses import dataclass
from enum import Enum

class FlowRegime(str, Enum):
    LAMINAR = "laminar"
    TURBULENT = "turbulent"

FIN_EFFICIENCY = 0.42

@dataclass
class ProcessorSpecs:
    die_length: float = 0.0525
    die_width: float = 0.045
    die_thickness: float = 0.0022
    tdp: float = 150.0
    R_jc: float = 0.1

    @property
    def die_area(self):
        return self.die_length * self.die_width


@dataclass
class HeatSinkSpecs:
    length: float = 0.09
    width: float = 0.116
    base_thickness: float = 0.0025
    num_fins: int = 60
    fin_thickness: float = 0.0008
    overall_height: float = 0.027

    @property
    def fin_height(self):
        return self.overall_height - self.base_thickness

    @property
    def fin_spacing(self):
        return (self.width - self.num_fins * self.fin_thickness) / (self.num_fins - 1)


@dataclass
class MaterialProperties:
    aluminum_k: float = 167.0
    tim_k: float = 4.0
    tim_thickness: float = 0.0001


@dataclass
class AirProperties:
    temperature: float = 25.0
    k: float = 0.0262
    nu: float = 1.57e-5
    pr: float = 0.71
    velocity: float = 1.0


class HeatSinkThermalModel:

    def __init__(self, p, hs, m, air):
        self.p = p
        self.hs = hs
        self.m = m
        self.air = air

    def R_tim(self):
        return self.m.tim_thickness / (self.m.tim_k * self.p.die_area)

    def R_cond(self):
        return self.hs.base_thickness / (self.m.aluminum_k * self.p.die_area)

    def convection(self):
        Sf = self.hs.fin_spacing
        Re = self.air.velocity * Sf / self.air.nu

        if Re < 2300:
            Nu = 1.86 * (Re * self.air.pr * (2 * Sf / self.hs.length)) ** (1 / 3)
            regime = FlowRegime.LAMINAR
        else:
            Nu = 0.023 * Re ** 0.8 * self.air.pr ** 0.3
            regime = FlowRegime.TURBULENT

        h = Nu * self.air.k / (2 * Sf)

        fin_area = (
            self.hs.num_fins
            * 2
            * self.hs.fin_height
            * self.hs.length
            * FIN_EFFICIENCY
        )

        base_area = (
            self.hs.length * self.hs.width
            - self.hs.num_fins * self.hs.fin_thickness * self.hs.length
        )

        A_total = fin_area + base_area
        R_conv = 1 / (h * A_total)

        return {
            "Re": Re,
            "Nu": Nu,
            "h": h,
            "A_total": A_total,
            "R_conv": R_conv,
            "regime": regime.value,
        }

    def solve(self):
        R_hs = self.R_cond() + self.convection()["R_conv"]
        R_total = self.p.R_jc + self.R_tim() + R_hs

        return {
            "junction_temperature_physical": self.air.temperature + self.p.tdp * R_total,
            "details": self.convection(),
        }
