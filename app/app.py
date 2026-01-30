from flask import Flask, request, jsonify
import numpy as np
import dataclasses

from models.thermal_model import *
from models.pgnn_infer import pgnn_predict

app = Flask(__name__)

def filter_dataclass_fields(cls, data: dict):
    """Filter only valid constructor arguments for a dataclass."""
    valid_keys = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in data.items() if k in valid_keys}

@app.route("/api/predict", methods=["POST"])
def predict():
    d = request.get_json()

    # Processor
    processor_data = filter_dataclass_fields(ProcessorSpecs, d.get("processor", {}))
    p = ProcessorSpecs(**processor_data)

    # Heat sink (ignore fin_height in JSON)
    # Heat sink (ignore fin_height in JSON)
    heat_sink_data = d.get("heat_sink", {})
# Only allow actual constructor fields
    valid_keys = {f.name for f in dataclasses.fields(HeatSinkSpecs)}
    heat_sink_data_filtered = {k: v for k, v in heat_sink_data.items() if k in valid_keys}
    hs = HeatSinkSpecs(**heat_sink_data_filtered)


    # Material
    material_data = filter_dataclass_fields(MaterialProperties, d.get("materials", {}))
    m = MaterialProperties(**material_data)

    # Air
    air_data = filter_dataclass_fields(AirProperties, d.get("air", {}))
    air = AirProperties(**air_data)

    # Physics-based model
    physics = HeatSinkThermalModel(p, hs, m, air)
    base = physics.solve()

    # Construct input for PGNN (hs.fin_height is computed property)
    x = np.array([[  
        p.die_length,
        p.die_width,
        p.tdp,
        air.velocity,
        hs.num_fins,
        hs.fin_height,
        m.aluminum_k
    ]])

    delta = pgnn_predict(x)

    return jsonify({
        "Tj_physics": float(base["junction_temperature_physical"]),
        "Tj_corrected": float(base["junction_temperature_physical"] + delta),
        "delta_T": float(delta),
        "details": base["details"]
    })

if __name__ == "__main__":
    app.run(debug=True)
