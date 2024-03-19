import math
from argparse import Namespace
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from lru.architectures import DLRU, DLRUConfig
from tqdm import tqdm
import nonlinear_benchmarks


if __name__ == "__main__":

    cfg = {
        "d_model": 5,
        "d_state": 16,
        "n_layers": 3,
        "ff": "MLP",
        "max_phase": math.pi
    }
    cfg = Namespace(**cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch.set_num_threads(10)
    #torch.set_float32_matmul_precision("high")

    train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
    sampling_time = train_val.sampling_time #in seconds
    u_train, y_train = train_val #or train_val.u, train_val.y
    u_test, y_test = test        #or test.u,      test.y
    
    u_train = u_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    u_test = u_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    

    scaler_u = StandardScaler()
    u = scaler_u.fit_transform(u_train)

    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y_train)


    config = DLRUConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, max_phase=cfg.max_phase)
    model = DLRU(1, 1, config)

    #model.configure_optimizer(learning_rate=1e-4, weight_decay=1e-2)
    #model = torch.compile(model) # useful?

    u = torch.tensor(u).unsqueeze(0).float()
    y = torch.tensor(y).unsqueeze(0).float()

    #u = u[:, :100_000, :]
    #y = y[:, :100_000, :]
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)

    LOSS = []
    # Train
    for itr in tqdm(range(100_000)):

        y_pred = model(u)
        loss = torch.nn.functional.mse_loss(y, y_pred)

        loss.backward()
        opt.step()

        opt.zero_grad()
        if itr % 100 == 0:
            print(loss.item())
        LOSS.append(loss.item())

    checkpoint = {
        'scaler_u': scaler_u,
        'scaler_y': scaler_y,
        'model': model.state_dict(),
        'LOSS': np.array(LOSS),
    }

    torch.save(checkpoint, "ckpt.pt")
