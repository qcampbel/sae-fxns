import torch
import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from quinny_sae_fxns.utils import print_hf_file_sizes, get_device


PRETRAINED_MODELS = {
    'facebook/galactica-125m',
    'facebook/galactica-6.7b',
    'osunlp/LlaSMol-Mistral-7B',
    'osunlp/LlaSMol-Galactica-6.7B',
    'osunlp/LlaSMol-Llama2-7B',
    'osunlp/LlaSMol-CodeLlama-7B',
}

BASE_MODELS = {
    'osunlp/LlaSMol-Mistral-7B': 'mistralai/Mistral-7B-v0.1',
    'osunlp/LlaSMol-Galactica-6.7B': 'facebook/galactica-6.7b',
    'osunlp/LlaSMol-Llama2-7B': 'meta-llama/Llama-2-7b-hf',
    'osunlp/LlaSMol-CodeLlama-7B': 'codellama/CodeLlama-7b-hf',
}

def get_context_galatica(data, task): # CURRENTLY NOT USED
    """
    Preprocesses inputs for Galactica model.
    """
    contexts = []
    if task == 'name_conversion-i2s':
        contexts = data['input']  # use SMolInstruct's default input texts #TODO: edit
    elif task == 'name_conversion-s2i':
        inputs = data['raw_input']
        if isinstance(inputs, str):
            inputs = [inputs]
        for smiles in inputs:
            context = (f"\n\nConvert the following SMILES notation <SMILES> {smiles}" 
                       " </SMILES> into its IUPAC nomenclature. ")
            contexts.append(context)
    else:
        raise NotImplementedError(f"Yet to be implemented. Task: {task}")
    if len(contexts)==1:
        return contexts[0]
    return contexts

def prepare_inputs(data, model_name, task='name_conversion-i2s'): # CURRENTLY NOT USED
    """
    Prepares input data according to the model's requirements.
    Note: designed for SMolInstruct dataset
    """
    # if isinstance(data, str):
    #     data = [data]
    
    if model_name.startswith('facebook/galactica'):
        updated_inputs = get_context_galatica(data, task)
    else:
        raise NotImplementedError(f"Yet to be implemented. Model: {model_name}")
    
    data['original_input'] = data['input']
    data['input'] = updated_inputs
    return data

def load_tokenizer_and_model(model_name, base_model_name=None, device=None):
    if model_name not in PRETRAINED_MODELS:
        raise NotImplementedError(f"This model is not supported: {model_name}")
    if base_model_name is None and model_name in BASE_MODELS: # for finetuned models only
        base_model_name = BASE_MODELS[model_name]
    if device is None:
        device = get_device()
    
    print_hf_file_sizes(model_name, repo_type='model') 
    if base_model_name: # only occurs with LlasMol, it's technically finetuned model
        print_hf_file_sizes(base_model_name, repo_type='model')
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name,torch_dtype=torch.bfloat16).to(device) 
        model = PeftModel.from_pretrained(base_model, model_name, torch_dtype=torch.bfloat16).to(device)
        model = model.merge_and_unload()

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

    tokenizer.padding_side = 'left'
    tokenizer.pad_token = '<pad>'
    tokenizer.sep_token = '<unk>'
    tokenizer.cls_token = '<unk>'
    tokenizer.mask_token = '<unk>'
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()

    return tokenizer, model
