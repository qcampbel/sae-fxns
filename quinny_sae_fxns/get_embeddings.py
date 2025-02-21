import torch

from config import Config
from model import prepare_inputs

def get_hidden_states(inputs, layer_num, model_name, model, tokenizer):
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

    for text in data['input']:
        #tokenized_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(config.device)
        tokenized_input = tokenizer(text, return_tensors="pt").to(config.device)
        input_ids = tokenized_input['input_ids']
        token_ids.append(input_ids.cpu().numpy().tolist())
        
        with torch.no_grad():
            model_outputs = model.generate(
                input_ids, 
                output_hidden_states=True,
                return_dict_in_generate=True,
                #max_new_tokens=config.max_new_length,
            )
        
        # extract activations from hidden states of the chosen layer
        hidden_states = model_outputs.hidden_states
        hidden_layer_state = hidden_states[0][config.transformer_layer_num]
        layer_acts = hidden_layer_state[0]
        # if padding: # so all activations have same length
        #     act_len = len(layer_acts)
        #     target_len = config.max_context_length
        #     if act_len > target_len:
        #         layer_acts = layer_acts[:target_len]
        #     elif act_len < target_len:
        #         padding = torch.zeros(target_len - act_len, layer_acts.shape[1])
        #         layer_acts = torch.cat([padding, layer_acts], dim=0)

        activations.append(layer_acts.cpu().numpy().tolist())
        
        generated_text = tokenizer.decode(model_outputs.sequences[0], skip_special_tokens=False)
        generated_outputs.append(generated_text)  # Save the generated text
        
    data = data.add_column('activations', activations) 
    data = data.add_column('generated_outputs', generated_outputs)
    data = data.add_column('token_ids', token_ids)

    return data



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

        embeddings = get_hidden_states(
            batch, config.transformer_layer_num, config.model_name
        )
        # if config.transformer_layer_num == -1:
        #     embeddings = hidden_states[-2] 
        # else:
        #     embeddings = hidden_states[config.transformer_layer_num]
        
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