from dataclasses import dataclass


@dataclass
class Config:
    device = "mps"
    data_name="s2i"

    # transformer model
    model_name = "facebook/galactica-125m"
    short_name = "galactica-125m"
    d_model = 768
    max_context_length = 128
    #max_new_length = 128 # IUPAC names require many tokens, up to 150 to be safe
    vocab_size = 50000
    save_acts_dir = f"saved_activation_datasets/activations_{short_name}_{data_name}"

    # sae model
    R = 12
    d_features = R * d_model
    transformer_layer_num = 7 # this tells us which transformer MLP layer to train on

    # autoencoder training
    a_batch_size = 8
    a_lr = 1e-6
    a_num_epochs = 20
    a_out_path = f"sae-checkpoints/autoencoder_L{transformer_layer_num}_d{d_features}.pt"
    lambda_reg = 3e-3
    num_neuron_resampling = d_features // 2
    save_activations = True
    num_act_samples = 50