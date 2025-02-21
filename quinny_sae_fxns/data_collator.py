import torch
#from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
 
def get_tokenizer_dataloader(dataset, tokenizer, batch_size, mlm=False, shuffle=True):
    """
    Creates a DataLoader for a specific dataset or task, depending on the input.
    
    Args:
        dataset (Dataset): The dataset from Huggingface
        tokenizer (AutoTokenizer): The tokenizer to use for processing the text.
        batch_size (int): The batch size for the DataLoader.
        mlm (bool): Whether to use Masked Language Modeling or Next-Token prediction (default is False for next-token).
        shuffle (bool): Whether to shuffle the data (default is True).

    Returns:
        torch.utils.data.DataLoader: The DataLoader for the specified dataset or task.
    """
    def tokenize_function(examples):
        return tokenizer(examples['input']) #, padding='max_length', truncation=True)
    
    tokenized_data = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm  # Set to False for next-token prediction (default behavior)
    )
    data_loader = DataLoader(
        tokenized_data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    )

    return data_loader



def get_sae_dataloader(dataset, batch_size, shuffle=True):
    """
    Creates a DataLoader for a specific dataset or task, depending on the input.
    
    Args:
        dataset (Dataset): The dataset from Huggingface
        tokenizer (AutoTokenizer): The tokenizer to use for processing the text.
        batch_size (int): The batch size for the DataLoader.
        mlm (bool): Whether to use Masked Language Modeling or Next-Token prediction (default is False for next-token).
        shuffle (bool): Whether to shuffle the data (default is True).

    Returns:
        torch.utils.data.DataLoader: The DataLoader for the specified dataset or task.
    """
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=mlm  # Set to False for next-token prediction (default behavior)
    # )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    )

    return data_loader
