from operator import index
from transformers import BertModel, BertTokenizer, BertConfig
import torch

enc = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

all_encoder_layers, pooled_output = loaded_model(*dummy_input)
print(all_encoder_layers)
print(pooled_output)

tokens_tensor = torch.tensor([indexed_tokens, indexed_tokens, indexed_tokens, indexed_tokens, indexed_tokens])
segments_tensors = torch.tensor([segments_ids, segments_ids, segments_ids, segments_ids, segments_ids])
print(loaded_model(tokens_tensor, segments_tensors))

