from typing import Any


# MODEL_PARAMETERS
ATM: float = 1.0  # [Pa], atmospheric pressure

# region case A
case_A: dict[str, Any] = {
    "WIDTH": 2.8,  # [m]
    "HEIGHT": 1.2,  # [m]
    "INITIAL_PRESSURE": ATM,  # [Pa]
    "INITIAL_SATURATION": 0.8,  # [-], initial saturation. Domain filled with water.
    # "INJECTION_RATE": 1.7e-7 / _co2["density"],  # 1.7x10^- 7kg/s
    "WELL_SIZE": 0.02,  # [m]
    "WELL_1_POS": (0.9, 0.3),  # [m]
    "WELL_2_POS": (1.7, 0.68),  # [m], slightly lower so it's fully inside facies 4.
    "MAX_CAP_PRESS": 9.5e4,  # [Pa], upper limit on capillary pressure.
    "PERMEABILITY": {  # Permeability in [m^2]
        "Facies 1": 4e-11,
        "Facies 2": 5e-10,
        "Facies 3": 1e-9,
        "Facies 4": 2e-9,
        "Facies 5": 4e-9,
        "Facies 6": 1e-8,
        "Facies 7": 1e-30,  # Epsilon to avoid division by zero when calculating face transmissibilities.
    },
    "POROSITY": {  # Porosity in [-]
        "Facies 1": 0.44,
        "Facies 2": 0.43,
        "Facies 3": 0.44,
        "Facies 4": 0.45,
        "Facies 5": 0.43,
        "Facies 6": 0.46,
        "Facies 7": 1e-20,  # Epsilon to avoid ill-defined problem. Funker også med 0 her nå.
    },
    "ENTRY_PRESSURE": {  # Entry pressures in [Pa]
        "Facies 1": 1500.0,
        "Facies 2": 300.0,
        "Facies 3": 100.0,
        "Facies 4": 25.0,
        "Facies 5": 10.0,
        "Facies 6": 1.0,
        "Facies 7": 1e-20,  # Epsilon to avoid ill-defined problem.
    },
}
