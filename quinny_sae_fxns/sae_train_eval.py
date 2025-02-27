import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm

from .autoencoder import SparseAutoencoder
from .config import Config
from .get_embeddings import get_hidden_states, get_activations

import h5py
import torch
import gc
import numpy as np
from tqdm import tqdm
from datasets import Dataset

def extract_activations_and_save_hdf5(data, model, tokenizer, config, save_every=1000):
    """
    Extracts activations, token IDs, and generated token IDs, then stores them in an HDF5 file.
    - Periodically writes to disk every `save_every` samples to prevent memory overload.
    - Clears CPU memory frequently.
    - Decodes token IDs into text **only** when converting to Hugging Face dataset.
    """

    hdf5_path = f"{config.embeddings_dir}/activations.h5"
    with h5py.File(hdf5_path, "w") as f:
        activations_grp = f.create_group("activations")  # Group for activations
        token_ids_grp = f.create_group("token_ids")  # Group for input token IDs
        generated_ids_grp = f.create_group("generated_token_ids")  # Group for generated token IDs

        all_activations, all_token_ids, all_generated_ids = [], [], []

        with torch.no_grad():
            for i, text in enumerate(tqdm(data['input'], desc="Generating embeddings", unit="inputs")):
                # Tokenize and move to GPU
                tokenized_input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=config.max_context_length).to(config.device)
                input_ids = tokenized_input["input_ids"]

                # Generate model outputs
                model_outputs = model.generate(
                    input_ids,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    max_new_tokens=config.max_new_length,
                )

                # Extract hidden states
                hidden_states = model_outputs.hidden_states
                hidden_layer_state = hidden_states[0][config.transformer_layer_num]
                layer_acts = hidden_layer_state[0].cpu().numpy()  # Convert activation to NumPy

                # Store data in memory temporarily
                all_activations.append(layer_acts)
                all_token_ids.append(input_ids.cpu().numpy())  # Store tokenized input
                all_generated_ids.append(model_outputs.sequences[0].cpu().numpy())  # Store generated token IDs

                # Periodically write to disk and clear memory
                if (i + 1) % save_every == 0:
                    print(f"Flushing batch {i + 1} to disk...")

                    activations_grp.create_dataset(f"batch_{i}", data=np.array(all_activations, dtype=np.float32))
                    token_ids_grp.create_dataset(f"batch_{i}", data=np.array(all_token_ids, dtype=np.int32))
                    generated_ids_grp.create_dataset(f"batch_{i}", data=np.array(all_generated_ids, dtype=np.int32))

                    # Clear memory
                    all_activations.clear()
                    all_token_ids.clear()
                    all_generated_ids.clear()
                    gc.collect()
                    torch.cuda.empty_cache()

        # Save any remaining data
        if all_activations:
            print("Saving final batch to disk...")
            activations_grp.create_dataset(f"batch_final", data=np.array(all_activations, dtype=np.float32))
            token_ids_grp.create_dataset(f"batch_final", data=np.array(all_token_ids, dtype=np.int32))
            generated_ids_grp.create_dataset(f"batch_final", data=np.array(all_generated_ids, dtype=np.int32))

        print(f"Saved activations, token IDs, and generated token IDs to {hdf5_path}.")

    return hdf5_path  # Return path for further processing


def add_hdf5_to_dataset(original_dataset, hdf5_path, tokenizer):
    """
    Loads activations, token IDs, and generated token IDs from HDF5,
    decodes generated texts, and adds them to the original dataset.
    """

    with h5py.File(hdf5_path, "r") as f:
        activations, token_ids, generated_ids = [], [], []

        # Read all stored batches
        for batch_name in f["activations"]:
            activations.extend(f["activations"][batch_name][:])  # Load activations
            token_ids.extend(f["token_ids"][batch_name][:])  # Load input token IDs
            generated_ids.extend(f["generated_token_ids"][batch_name][:])  # Load generated token IDs

        # Decode generated token IDs into text
        generated_texts = [tokenizer.decode(g_ids, skip_special_tokens=False) for g_ids in generated_ids]

    # Ensure lengths match
    assert len(original_dataset) == len(activations) == len(token_ids) == len(generated_texts), "Data length mismatch!"

    # Add new fields to the original dataset
    updated_dataset = original_dataset.add_column("activations", activations)
    updated_dataset = updated_dataset.add_column("token_ids", token_ids)
    updated_dataset = updated_dataset.add_column("generated_texts", generated_texts)

    return updated_dataset



def train_sae_on_activations(activations, autoencoder_model, optimizer, config):
    """
    This function trains the Sparse Autoencoder (SAE) on precomputed activations.
    Since it's unsupervised, activations are compared against themselves.
    """
    
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=config.a_batch_size, shuffle=True)

    losses = []
    recon_losses=[]
    reg_losses = []

    autoencoder_model.train()

    for epoch in range(config.a_num_epochs):
        for i, (activation_batch,) in enumerate(dataloader):
            optimizer.zero_grad()

            recon_batch, recon_loss, reg_loss = autoencoder_model(activation_batch, compute_loss=True)

            total_loss = recon_loss + config.lambda_reg * reg_loss
            total_loss.backward()
            optimizer.step()

            losses.append(total_loss.item())
            recon_losses.append(recon_loss.item())
            reg_losses.append(reg_loss.item())

            #if i % 20 == 0:  # Print loss every 20 steps
        print(f'Epoch {epoch}/{config.a_num_epochs}, Batch #{i}, Loss: {total_loss.item():.3f}, Recon Loss: {recon_loss.item():.3f}, Reg Loss: {reg_loss.item():.3f}')

    return losses, recon_losses, reg_losses


# OLD SAE stuff - TODO: refactor to separate train/evaluate OR combine all SAE fxns together
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

