# Hugging Face Gemma Recipes

![repository thumbnail](../assets/thumbnail.png)

🤗💎 Welcome! This repository contains *minimal* recipes to get started quickly with the Gemma family of models.

> [!Note]
> Fine tune Gemma 3n 2B on a Free Colab Notebook: <a href="https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/fine_tune_gemma3n_on_t4.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
>
> Fine tune Gemma 3n 4B on a Free Colab Notebook: <a href="https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/Gemma3N_(4B)-Conversational.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
>
> Multimodal inference using Gemma 3n via pipeline <a href="https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/gemma3n_inference_via_pipeline.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>


## Getting Started

To quickly run a Gemma 💎 model on your machine, install the latest version of `timm` (for the vision encoder) and 🤗 `transformers` to run inference, or if you want to fine tune it.

```shell
$ pip install -U -q transformers timm
```

### Inference with pipeline

The easiest way to start using Gemma 3n is by using the pipeline abstraction in transformers:

```python
import torch
from transformers import pipeline

pipe = pipeline(
   "image-text-to-text",
   model="google/gemma-3n-E4B-it", # "google/gemma-3n-E4B-it"
   device="cuda",
   torch_dtype=torch.bfloat16
)

messages = [
   {
       "role": "user",
       "content": [
           {"type": "image", "url": "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/airplane.jpg"},
           {"type": "text", "text": "Describe this image"}
       ]
   }
]

output = pipe(text=messages, max_new_tokens=32)
print(output[0]["generated_text"][-1]["content"])
```

### Detailed inference with transformers

Initialize the model and the processor from the Hub, and write the `model_generation` function that takes care of processing the prompts and running the inference on the model.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "google/gemma-3n-e4b-it" # google/gemma-3n-e2b-it
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)

def model_generation(model, messages):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_len = inputs["input_ids"].shape[-1]

    inputs = inputs.to(model.device, dtype=model.dtype)

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=32, disable_compile=False)
        generation = generation[:, input_len:]

    decoded = processor.batch_decode(generation, skip_special_tokens=True)
    print(decoded[0])
```

And then using calling it with our specific modality:

#### Text only

```python
# Text Only

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the capital of France?"}
        ]
    }
]
model_generation(model, messages)
```

#### Interleaved with Audio

```python
# Interleaved with Audio

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech segment in English:"},
            {"type": "audio", "audio": "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/speech.wav"},
        ]
    }
]
model_generation(model, messages)
```

#### Interleaved with Image/Video

```python
# Interleaved with Image

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/airplane.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]
model_generation(model, messages)
```

## Inference

### Gemma 3n

#### Notebooks

* [Multimodal inference using Gemma 3n via pipeline](/notebooks/gemma3n_inference_via_pipeline.ipynb) <a href="https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/gemma3n_inference_via_pipeline.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## Fine Tuning

We include a series of notebook+scripts for fine tuning the models.

### Gemma 3n

#### RAG
* [Fine tuning Gemma 3n on RAG](/notebooks/Gemma_RAG.ipynb)

#### Notebooks

* [Fine tuning Gemma 3n 2B on free Colab T4](/notebooks/fine_tune_gemma3n_on_t4.ipynb) <a href="https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/fine_tune_gemma3n_on_t4.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
* [Fine tuning Gemma 3n 4B with Unsloth on free Colab T4](/notebooks/Gemma3N_(4B)-Conversational.ipynb) <a href="https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/Gemma3N_(4B)-Conversational.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
* [Fine tuning Gemma 3n on audio](/notebooks/fine_tune_gemma3n_on_audio.ipynb)
* [Fine tuning Gemma 3n on images using TRL](/scripts/ft_gemma3n_image_trl.py)
* [Fine tuning Gemma 3n on GUI Grounding](/notebooks/Gemma_3n_GUI_Finetune.ipynb)

#### Scripts

* [Fine tuning Gemma 3n on images (script)](/scripts/ft_gemma3n_image_vt.py)
* [Fine tuning Gemma 3n on audio (script)](/scripts/ft_gemma3n_audio_vt.py)

### Gemma 3
  
* [Reinforement Learning (GRPO) on Gemma 3 with Unsloth and TRL](/notebooks/Gemma3_(1B)-GRPO.ipynb) <a href="https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/Gemma3_(1B)-GRPO.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
* [Vision fine tuning Gemma 3 4B with Unsloth](/notebooks/Gemma3_(4B)-Vision.ipynb) <a href="https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/Gemma3_(4B)-Vision.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
* [Conversational fine tuning Gemma 3 4B with Unsloth](/notebooks/Gemma3_(4B).ipynb) <a href="https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/Gemma3_(4B).ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

Before fine-tuning the model, ensure all dependencies are installed:

```bash
$ pip install -U -q -r requirements.txt
```

✨ **Bonus:** We've also experimented with adding **object detection** 🔍 capabilities to Gemma 3. You can explore that work in [this dedicated repo](https://github.com/ariG23498/gemma3-object-detection).

