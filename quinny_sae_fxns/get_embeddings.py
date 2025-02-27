import gc
import numpy as np
import os

import h5py
import torch
from datasets import Dataset
from tqdm import tqdm

from .config import Config
from .model import prepare_inputs


def extract_activations_and_save_hdf5(data, model, tokenizer, config, save_every=1000):
    """
    Extracts activations, token IDs, and generated token IDs, then stores them in an HDF5 file.
    - Periodically writes to disk every `save_every` samples to prevent memory overload.
    - Clears CPU memory frequently.
    - Decodes token IDs into text **only** when converting to Hugging Face dataset.
    """
    os.makedirs(config.embeddings_dir, exist_ok=True)
    hdf5_path = os.path.join(config.embeddings_dir, "activations.h5")
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

                    activations_grp.create_dataset(f"batch_{i}", data=np.array(all_activations, dtype=np.float16))
                    token_ids_grp.create_dataset(f"batch_{i}", data=np.array(all_token_ids, dtype=np.int16))
                    vlen_int_dtype = h5py.special_dtype(vlen=np.int16) # variable length of generated tokens
		    generated_ids_grp.create_dataset(f"batch_{i}", data=np.array(all_generated_ids, dtype=vlen_int_dtype))

                    # Clear memory
                    all_activations.clear()
                    all_token_ids.clear()
                    all_generated_ids.clear()
                    gc.collect()
                    torch.cuda.empty_cache()

        # Save any remaining data
        if all_activations:
            print("Saving final batch to disk...")
            activations_grp.create_dataset(f"batch_final", data=np.array(all_activations, dtype=np.float16))
            token_ids_grp.create_dataset(f"batch_final", data=np.array(all_token_ids, dtype=np.int16))
            generated_ids_grp.create_dataset(f"batch_final", data=np.array(all_generated_ids, dtype=vlen_int_dtype))

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



def get_hidden_states(model_outputs, layer_num, model_name):
    if model_name.startswith('facebook/galactica'):
        hidden_states = model_outputs.hidden_states
        hidden_layer_state = hidden_states[0][layer_num+1]
        layer_acts = hidden_layer_state[0]
    else:
        raise NotImplementedError(f"haven't implemented for model {model_name}")
    return layer_acts

def add_padding(tensor, target_len):
    act_len = len(layer_acts)
    target_len = config.max_context_length
    if act_len > target_len:
        layer_acts = layer_acts[:target_len]
    elif act_len < target_len:
        padding = torch.zeros(target_len - act_len, layer_acts.shape[1])
        layer_acts = torch.cat([padding, layer_acts], dim=0)
    pass

def extract_activations_and_add_to_dataset(data, model, tokenizer, config, padding=True):
    """
    This function computes activations, generates model outputs, and adds all of them (activations, token_ids, and generated outputs)
    to the dataset.
    - Activations are taken from the hidden states of the transformer model.
    - Generated text is produced from the model's generation process (not true model output).
    - Token IDs are the input text tokenized into model-specific IDs.
    """
    activations = []
    generated_outputs = []
    token_ids = []

    with torch.no_grad():
        for i, text in enumerate(tqdm(data['input'], desc="Generating and extracting embeddings", unit="inputs")):
            tokenized_input = tokenizer(
                text, return_tensors="pt", padding='max_length', truncation=True, max_length=config.max_context_length
            ).to(config.device)
            #tokenized_input = tokenizer(text, return_tensors="pt").to(config.device)
            input_ids = tokenized_input['input_ids']
            token_ids.append(input_ids.cpu().numpy().tolist())
            
            model_outputs = model.generate(
                input_ids, 
                output_hidden_states=True,
                return_dict_in_generate=True,
                max_new_tokens=config.max_new_length,
            )
            
            # extract activations from hidden states of the chosen layer
            hidden_states = model_outputs.hidden_states
            hidden_layer_state = hidden_states[0][config.transformer_layer_num]
            layer_acts = hidden_layer_state[0]
            
            activations.append(layer_acts.cpu().numpy().tolist())   
            generated_text = tokenizer.decode(model_outputs.sequences[0].cpu(), skip_special_tokens=False)
            generated_outputs.append(generated_text)  # Save the generated text

            # free up memory
            del tokenized_input, model_outputs, hidden_states, hidden_layer_state, layer_acts
            torch.cuda.empty_cache()
            
            if i % 100 == 0:  # Print progress every 100 steps
                tqdm.write(f"Processed {i}/{len(data['input'])} inputs...")
        
    data = data.add_column('activations', activations) 
    data = data.add_column('generated_outputs', generated_outputs)
    data = data.add_column('token_ids', token_ids)

    return data


def get_hidden_states_OLD(inputs, layer_num, model_name, model, tokenizer):
    # note: it may vary based on transformers' hidden state structure 
    with torch.no_grad():
        model_outputs = model.generate(
            **inputs, 
            output_hidden_states=True,
            return_dict_in_generate=True
        )
    hidden_states = model_outputs.hidden_states
    hidden_layer_state = hidden_states[0][layer_num+1] # galactica structure
    return hidden_layer_state

def get_activations(
        data_loader, 
        use_autoencoder=True, 
        autoencoder=None, 
        config=None
    ):
    # useful for creating plots of both SAE & transformer embeddings
    if not config:
        config = Config()
    activations = []

    for batch in data_loader:

        embeddings = get_hidden_states_OLD(
            batch, config.transformer_layer_num, config.model_name
        )
        batch_size, context_length, embedding_dim = embeddings.shape
        random_idxs = torch.randint(0, context_length, (batch_size,))
        random_embeddings = embeddings[torch.arange(batch_size), random_idxs]

        if use_autoencoder:
            if autoencoder == None:
                print("Error: 'use_autoencoder=True' is set, and 'autoencoder' returns None when it's required.")
            # if it's SAE, then get its own embeddings which is trained on transformer's embeddings
            autoencoder.eval()

            with torch.no_grad():
                random_embeddings = autoencoder.encode(random_embeddings)

        activations.append(random_embeddings)

    activations = torch.cat(activations, dim=0)  # flatten along batch dimension, Shape: (total_batches * batch_size, embedding_dim)
    activation_densities = (activations > 0).float().mean(dim=0) # get positive hits as 1s then average out across batches
    #activation_densities = activations * (activations > 0).float()
    #activation_densities = activations[activations > 1e9]

    activation_densities += 1e-8 # to avoid log(0)
    log_activation_densities = activation_densities.log10().cpu().tolist() # to clearly observe sparisty between 0-1
    #activation_densities = activation_densities.cpu().tolist()
    
    return log_activation_densities
    #return activation_densities
