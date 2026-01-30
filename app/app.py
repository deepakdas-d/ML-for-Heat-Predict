from flask import Flask, request, jsonify
import numpy as np
import dataclasses

from models.thermal_model import *
from models.pgnn_infer import pgnn_predict

app = Flask(__name__)

# -------------------------------
# Helper to filter valid dataclass fields
# -------------------------------
def filter_dataclass_fields(cls, data: dict):
    """Filter only valid constructor arguments for a dataclass."""
    valid_keys = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in data.items() if k in valid_keys}

# -------------------------------
# Physics-only analysis
# -------------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():
    d = request.get_json()

    # Processor
    processor_data = filter_dataclass_fields(ProcessorSpecs, d.get("processor", {}))
    p = ProcessorSpecs(**processor_data)

    # Heat sink
    heat_sink_data = filter_dataclass_fields(HeatSinkSpecs, d.get("heat_sink", {}))
    hs = HeatSinkSpecs(**heat_sink_data)

    # Material
    material_data = filter_dataclass_fields(MaterialProperties, d.get("materials", {}))
    m = MaterialProperties(**material_data)

    # Air
    air_data = filter_dataclass_fields(AirProperties, d.get("air", {}))
    air = AirProperties(**air_data)

    try:
        result = HeatSinkThermalModel(p, hs, m, air).solve()
        return jsonify({
            "Tj_physics": float(result.get("junction_temperature_physical", 0.0)),
            "Tj_excel": float(result.get("junction_temperature_excel", result.get("junction_temperature_physical", 0.0))),
            "details": result.get("details", {})
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# PGNN-corrected prediction
# -------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    d = request.get_json()

    # Processor
    p = ProcessorSpecs(**filter_dataclass_fields(ProcessorSpecs, d.get("processor", {})))

    # Heat sink (ignore fin_height in JSON)
    heat_sink_data = d.get("heat_sink", {})
    heat_sink_filtered = {k: v for k, v in heat_sink_data.items() if k in {f.name for f in dataclasses.fields(HeatSinkSpecs)}}
    hs = HeatSinkSpecs(**heat_sink_filtered)

    # Material
    m = MaterialProperties(**filter_dataclass_fields(MaterialProperties, d.get("materials", {})))

    # Air
    air = AirProperties(**filter_dataclass_fields(AirProperties, d.get("air", {})))

    # Physics-based model
    physics = HeatSinkThermalModel(p, hs, m, air)
    base = physics.solve()

    # Construct input for PGNN
    x = np.array([[  
        p.die_length,
        p.die_width,
        p.tdp,
        air.velocity,
        hs.num_fins,
        hs.fin_height,   # computed property
        m.aluminum_k
    ]])

    delta = pgnn_predict(x)

    return jsonify({
        "Tj_physics": float(base["junction_temperature_physical"]),
        "Tj_corrected": float(base["junction_temperature_physical"] + delta),
        "delta_T": float(delta),
        "details": base["details"]
    })

# -------------------------------
# Default API (no input, uses defaults)
# -------------------------------
@app.route("/api/default", methods=["GET"])
def default():
    p = ProcessorSpecs()
    hs = HeatSinkSpecs()
    m = MaterialProperties()
    air = AirProperties()

    physics = HeatSinkThermalModel(p, hs, m, air)
    base = physics.solve()

    return jsonify({
        "Tj_physics": float(base.get("junction_temperature_physical", 0.0)),
        "Tj_excel": float(base.get("junction_temperature_excel", base.get("junction_temperature_physical", 0.0))),
        "details": base.get("details", {})
    })


# -------------------------------
# Run Flask
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
