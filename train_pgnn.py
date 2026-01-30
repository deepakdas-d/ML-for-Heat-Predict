import numpy as np
import torch
from models.pgnn import HeatSinkPGNN
from models.thermal_model import *

def generate_data(n=2000):
    X, y = [], []

    for _ in range(n):
        p = ProcessorSpecs(
            die_length=np.random.uniform(0.03, 0.06),
            die_width=np.random.uniform(0.03, 0.06),
            tdp=np.random.uniform(50, 200),
        )
        hs = HeatSinkSpecs(num_fins=np.random.randint(20, 80))
        air = AirProperties(velocity=np.random.uniform(0.5, 4.0))
        m = MaterialProperties()

        model = HeatSinkThermalModel(p, hs, m, air)
        Tj = model.solve()["junction_temperature_physical"]

        bias = np.random.uniform(-5, 5)

        X.append([
            p.die_length, p.die_width, p.tdp,
            air.velocity, hs.num_fins,
            hs.fin_height, m.aluminum_k
        ])
        y.append(-bias)

    return np.array(X), np.array(y)

X, y = generate_data()
mean, std = X.mean(0), X.std(0) + 1e-8
Xn = (X - mean) / std

model = HeatSinkPGNN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

Xn = torch.tensor(Xn, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

for i in range(2000):
    opt.zero_grad()
    loss = loss_fn(model(Xn), y)
    loss.backward()
    opt.step()
    if i % 200 == 0:
        print(i, loss.item())

torch.save({"model": model.state_dict(), "mean": mean, "std": std}, "pgnn.pt")
