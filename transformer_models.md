# [The Transformer Models](https://www.coursera.org/learn/generative-ai-with-llms/lecture/2T3Au/pre-training-large-language-models)    
There were three variance of the transformer model; **encoder-only, encoder-decoder models, and decode-only**. Each of these is trained on a different 
objective, and so learns how to carry out different tasks.

1- **Encoder-only models**: these models are also known as **Autoencoding models**, and they are **pre-trained using masked language modeling**. Here, tokens in 
the input sequence or randomly mask, and the training objective is to predict the mask tokens in order to reconstruct the original sentence. This is also called 
a denoising objective. Autoencoding models spilled bi-directional representations of the input sequence, meaning that the model has an understanding of the full 
context of a token and not just of the words that come before. Encoder-only models are ideally suited to task that benefit from this bi-directional contexts. 
You can use them to carry out sentence classification tasks, for example, sentiment analysis or token-level tasks like named entity recognition or word classification. 
**Some well-known examples of an autoencoder model are BERT and RoBERTa**.
  
2- **Decoder-only models**: Now, let's take a look at decoder-only or **autoregressive models**, which are pre-trained using **causal language modeling**. 
Here, the training objective is to **predict the next token based on the previous sequence of tokens**. Predicting the next token is sometimes called full 
language modeling by researchers. Decoder-based autoregressive models, **mask the input sequence and can only see the input tokens leading up to the token in question**. 
The model has no knowledge of the end of the sentence. The model then iterates over the input sequence one by one to predict the following token. **In contrast to the encoder 
architecture, this means that the context is unidirectional**. By learning to predict the next token from a vast number of examples, the model builds up a statistical 
representation of language. Models of this type make use of the decoder component off the original architecture without the encoder. **Decoder-only models are often 
used for text generation**, although larger decoder-only models show strong zero-shot inference abilities, and can often perform a range of tasks well. Well known 
examples of decoder-based autoregressive models are **GBT and BLOOM**.

3- **Encoder-Decoder models**: The final variation of the transformer model is **the sequence-to-sequence model** that uses both the encoder and decoder parts off 
the original transformer architecture. The exact details of the pre-training objective vary from model to model. **A popular sequence-to-sequence model T5**, pre-trains 
the encoder using span corruption, which masks random sequences of input tokens. Those mass sequences are then replaced with a unique Sentinel token, shown here as x. 
Sentinel tokens are special tokens added to the vocabulary, but do not correspond to any actual word from the input text. The decoder is then tasked with reconstructing 
the mask token sequences auto-regressively. The output is the Sentinel token followed by the predicted tokens. **You can use sequence-to-sequence models for translation, 
summarization, and question-answering**. They are generally useful in cases where you have a body of texts as both input and output. Besides T5, which you'll use in the labs 
in this course, another well-known encoder-decoder model is BART, not bird.

## Three methods for loading a trained LLM model for utilization

When you instantiate one of the three models (AutoConfig, AutoModel, or AutoTokenizer) you will directly create a class corresponding to the relevant architecture.   

1- AutoConfig:
Use AutoConfig when you need to customize the configuration parameters of a pre-trained model.
It allows you to modify settings such as the number of layers, hidden size, attention mechanism, etc.
You might use AutoConfig if you want to fine-tune a pre-trained model with specific architecture modifications.
```
from transformers import AutoConfig
# Example usage: Instantiate a BERT configuration
config = AutoConfig.from_pretrained("bert-base-uncased")
```
2- AutoModel:
Use AutoModel when you want to load the architecture of a pre-trained model for further use in downstream tasks.
This class loads the pre-trained weights along with the architecture, allowing you to perform various tasks like fine-tuning, feature extraction, etc.
You might use AutoModel when you need to build a custom model architecture using pre-trained weights as a starting point.
```
from transformers import AutoModel
# Example usage: Instantiate a BERT model
model = AutoModel.from_pretrained("bert-base-uncased")
```
3- AutoTokenizer:
Use AutoTokenizer when you need to tokenize input text for use with a pre-trained model.
Tokenization is the process of splitting input text into individual tokens or subwords that the model can process.
AutoTokenizer automatically selects the appropriate tokenizer for the specified pre-trained model.
You might use AutoTokenizer when preparing text data for input into a pre-trained model for tasks like classification, generation, etc.
```
from transformers import AutoTokenizer
# Example usage: Instantiate a BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

## Exploring Three Methods for Loading AutoModel
Read [Causal LLMs and Seq2Seq Architectures](https://heidloff.net/article/causal-llm-seq2seq/) to get a complete information about different models in LLM, i.e., Causal LLMs and Seq2Seq Architectures!

1- Masked Language Modeling (MLM) - encoder
```
from transformers import AutoTokenizer, AutoModelForMaskedLM
model_id="bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)
```

2- Causal Language Modeling (CLM) - decoder
```
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id="gpt2"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

3- Sequence-to-Sequence (Seq2Seq) - encoder and decoder

```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_id="google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
```
