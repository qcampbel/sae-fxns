import numpy as np

import torch
import tqdm

from autoencoder import SparseAutoencoder
from config import Config
from get_embeddings import get_hidden_states, get_activations


# SAE stuff - TODO: refactor to separate train/evaluate OR combine all SAE fxns together
class SAEWrapper:
    def __init__(self, config=None):
        # set up SAE
        if not config:
            self.config = Config()
        self.autoencoder = SparseAutoencoder(
            config.d_features,
            config.d_model,
        ).to(config.device)
        

    def train(self, train_loader_prop, config=None):
        if config is None:
            config = self.config
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=config.a_lr)
        
        losses = []
        recon_losses = []
        reg_losses = []
        activation_sets = []

        self.autoencoder.train()

        for epoch in range(config.a_num_epochs):
            running_loss = 0.0
            running_recon_loss = 0.0
            running_reg_loss = 0.0
            progress_bar = tqdm(enumerate(train_loader_prop), desc=f"Epoch {epoch + 1}/{config.a_num_epochs}")

            for i, batch in progress_bar:
                # grab activations from RT2
                hidden_states = get_hidden_states(batch)
                if config.transformer_layer_num == -1:
                    x_embedding = hidden_states[-2] 
                else:
                    x_embedding = hidden_states[config.transformer_layer_num]

                
                optimizer.zero_grad()
                _, recon_loss, reg_loss = self.autoencoder(x_embedding, compute_loss=True)
                reg_loss = reg_loss * config.lambda_reg # regularization loss
                loss = recon_loss + reg_loss

                loss.backward()
                optimizer.step()
                self.autoencoder.normalize_decoder_weights() # ??

                losses.append(loss.item())
                recon_losses.append(recon_loss.item())
                reg_losses.append(reg_loss.item())

                running_loss += loss.item()
                running_recon_loss += recon_loss.item()
                running_reg_loss += reg_loss.item()
                progress_bar.set_postfix({
                    "loss": running_loss / (progress_bar.n + 1),
                    "recon_loss": running_recon_loss / (progress_bar.n + 1),
                    "reg_loss": running_reg_loss / (progress_bar.n + 1),
                })
            
            if config.save_activations: # save SAE's neuron activations
                print('Saving SAE activations...')
                sae_activation_densities = get_activations(
                    config.num_act_samples, use_autoencoder=True, autoencoder=self.autoencoder
                )
                activation_sets.append(sae_activation_densities)

        print(f"Finished training. Recon loss: {np.mean(recon_losses[-100:]):.3f}, Reg loss: {np.mean(reg_losses[-100:]):.3f}")
        results = {
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'recon_losses': recon_losses,
            'reg_losses' : reg_losses,
            'steps': len(losses),
            'epochs': config.a_num_epochs,
            'activation_sets': activation_sets if config.save_activations else None
        }
        return results

    def save(self, results, config=None):
        if config is None:
            config = self.config
        
        torch.save(
            {
                'model_state_dict': self.autoencoder.state_dict(),
                'optimizer_state_dict': results['optimizer_state_dict'],
                'losses': results['losses'],
                'recon_losses': results['recon_losses'],
                'reg_losses' : results['reg_losses'],
                'steps': results['steps'],
                'epochs': results['epochs'],
                'activation_sets': results['activation_sets'],
            }, config.a_out_path)
        print(f"Saved model to {config.a_out_path}")

    def evaluate(self, loader_prop):
        self.autoencoder.eval()
        progress_bar = tqdm(enumerate(loader_prop), desc="Calculating losses")

        with torch.no_grad():
            for i, batch in progress_bar:
                # grab activations from RT2
                hidden_states = get_hidden_states(batch)
                if self.config.transformer_layer_num == -1:
                    x_embedding = hidden_states[-2] 
                else:
                    x_embedding = hidden_states[self.config.transformer_layer_num]

                # evaluate how well it fits to RT2 embeddings/activations
                _, recon_loss, reg_loss = self.autoencoder(x_embedding, compute_loss=True)
                reg_loss = reg_loss * self.config.lambda_reg # regularization loss
                total_loss = recon_loss + reg_loss

        return total_loss.item(), recon_loss.item(), reg_loss.item()

