from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    device: str = field(default_factory=str)
    data_source: str = "osunlp/SMolInstruct"
    task: str = "name_conversion-s2i"
    split: str = "test"
    use_test_subset: bool = True

    # transformer model
    model_name: str = "facebook/galactica-125m"
    d_model: int = 768
    max_context_length: int = 128
    max_new_length: int = 128
    vocab_size: int = 50000
    transformer_layer_num: int = 7

    # autoencoder model training
    R: int = 4
    d_features: int = field(init=False)
    a_batch_size: int = 8
    a_lr: float = 1e-6
    a_num_epochs: int = 20
    lambda_reg: float = 3e-3
    
    num_act_samples: int = 50
    embeddings_dir: str = field(default_factory=str)

    def __post_init__(self):
        if torch.backends.mps.is_available():
            self.device="mps"
        elif torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"
        self.short_model_name = self.model_name.split("/")[-1]
        self.d_features = self.R * self.d_model
        self.num_neuron_resampling = self.d_features // 2
        self.sae_model_path = f"sae-checkpoints/autoencoder_L{self.transformer_layer_num}_d{self.d_features}.pt"
        if not self.embeddings_dir:
            self.embeddings_dir = f"saved_activation_datasets/activations_{self.short_model_name}_{self.task}_{self.split}"
