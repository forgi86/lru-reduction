from pathlib import Path
import torch
import numpy as np
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm
from lru.architectures import DLRU, DLRUConfig
import f16_utils
import wandb
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def get_config(config):
    global cfg
    cfg = config
    print(cfg)


if __name__ == "__main__":

    # not the suggested way to handle the config, but i wanna work in __main__ for the moment
    get_config() 

    torch.manual_seed(42)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(10)
    torch.set_float32_matmul_precision("high")
    
    n_u = 1
    n_y = 3

    if cfg.log_wandb:
        wandb.init(
            project="sysid-ss-bench",
            #name="run1",
            config=dict(cfg)
        )

    config = DLRUConfig(d_model=cfg.d_model,d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff)
    model = DLRU(n_u, n_y, config).to(device)

    ds_dict = f16_utils.load_data_dict()
    scaler_y, scaler_u = f16_utils.make_scalers(ds_dict["F16Data_FullMSine_Level5"])
    scaled_datasets = f16_utils.make_scaled_datasets(ds_dict, scaler_y, scaler_u)
    train_data = f16_utils.make_subsequence_datasets(scaled_datasets, subseq_len=cfg.seq_len)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    save_folder = Path("ckpt")
    save_folder.mkdir(exist_ok=True)
    LOSS = []
    # Train
    best_loss = np.inf
    for epoch in range(cfg.epochs):
        for itr, (batch_y, batch_u) in tqdm(enumerate(train_loader)):

            optimizer.zero_grad()
            
            batch_y = batch_y.to(device)
            batch_u = batch_u.to(device)

            batch_y_sim = model(batch_u)

            loss = torch.nn.functional.mse_loss(batch_y_sim[:, cfg.skip:, :], batch_y[:, cfg.skip:, :]) 

            loss_reg = torch.tensor(0.0, device=batch_u.device, dtype=batch_u.dtype)
            if cfg.reg_type == "modal" and cfg.reg_lambda > 0.0:
                for layer in range(cfg.n_layers):
                    lambdas_abs =  torch.exp(-torch.exp(model.blocks[layer].lru.nu_log))
                    loss_reg = loss_reg + cfg.reg_lambda*lambdas_abs.mean()

            elif cfg.reg_type == "hankel" and cfg.reg_lambda > 0.0:
                for layer in range(cfg.n_layers):
                    hankel_values = model.blocks[layer].lru.hankel_singular_values(cc=False)
                    loss_reg = loss_reg + cfg.reg_lambda*hankel_values.mean()

            elif cfg.reg_type == "hankel_cc" and cfg.reg_lambda > 0.0:
                for layer in range(cfg.n_layers):
                    hankel_values = model.blocks[layer].lru.hankel_singular_values(cc=True)
                    loss_reg = loss_reg + cfg.reg_lambda*hankel_values.mean()


            loss = loss + loss_reg

            #if epoch == 0 and itr == 1875:
            #    ad
            loss.backward()
            optimizer.step()

            if itr % 100 == 0:
                print(loss.item(), loss_reg.item())
                if cfg.log_wandb:
                    wandb.log({"loss": loss.item()})
                    if cfg.reg_lambda > 0:
                        wandb.log({"loss_reg": loss_reg.item()})

            if itr % 1000 == 0:
                if loss < best_loss:
                    best_loss = loss.item()
                    checkpoint = {
                    'scaler_u': scaler_u,
                    'scaler_y': scaler_y,
                    'model': model.state_dict(),
                    'LOSS': np.array(LOSS),
                    'cfg': cfg
                    }
                print("Saving model...")
                torch.save(checkpoint, save_folder / f"{cfg.out_name}.pt")

            LOSS.append(loss.item())

       
    checkpoint = {
        'scaler_u': scaler_u,
        'scaler_y': scaler_y,
        'model': model.state_dict(),
        'LOSS': np.array(LOSS),
        'cfg': cfg
    }

    torch.save(checkpoint, save_folder / f"{cfg.out_name}_last.pt")

