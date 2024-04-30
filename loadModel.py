import transformers as ts
from transformers import LlavaConfig, LlavaPreTrainedModel, AutoModel, AutoModelForCausalLM, Cache
import torch.nn as nn
from typing import List, Union

# from transformers import BitsAndBytesConfig

# # Load base model
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit= True,
#     bnb_4bit_quant_type= "nf4",
#     bnb_4bit_compute_dtype= torch.float16,
#     bnb_4bit_use_double_quant= False,
# )

model = ts.LlavaForConditionalGeneration.from_pretrained(
    "mohammadmahdinouri/llava-gemma-4bit", 
    # quantization_config=bnb_config, 
    device_map="mps",
    token="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").to("mps")

# print(model)