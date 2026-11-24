import numpy as np
from blast import models


def _profile():
    t = np.arange(0, 86400 + 1, 900.0)
    return {
        "Time_s": t,
        "SOC": 0.5 + 0.4 * np.sin(np.linspace(0, 2 * np.pi, t.size)),
        "Temperature_C": np.full(t.size, 25.0),
    }


def test_constant_input_without_energy_conservation_degrades_slower():
    """Without conserved throughput each repeat moves less charge as capacity falls, so fade must be smaller."""
    conserved = models.Nmc111_Gr_Kokam75Ah_Battery()
    conserved.simulate_battery_life(_profile(), threshold_time=1, is_constant_input=True, is_conserve_energy_throughput=True)
    scaled = models.Nmc111_Gr_Kokam75Ah_Battery()
    scaled.simulate_battery_life(_profile(), threshold_time=1, is_constant_input=True, is_conserve_energy_throughput=False)
    assert scaled.outputs["q"][-1] > conserved.outputs["q"][-1]
