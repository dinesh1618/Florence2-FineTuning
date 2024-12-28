# Fine-Tuning Microsoft Florence-2-Large Model

This repository provides a guide for fine-tuning the `microsoft/florence-2-large` model on a custom dataset. The model, designed for large-scale vision-language tasks, is adapted here for specific downstream applications.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Fine-Tuning](#fine-tuning)
- [Usage](#usage)
- [License](#license)

## Overview

This project demonstrates how to fine-tune the `microsoft/florence-2-large` model. The process includes dataset preparation, model configuration, and training. Post fine-tuning, the model can be deployed for image-text tasks like captioning, retrieval, or classification.

## Installation

Install the necessary dependencies before running the notebook:

```bash
pip install transformers datasets torch torchvision accelerate bitsandbytes -q
```

## Dataset Preparation

1. **Dataset Loading**:
   - Use a dataset suitable for vision-language tasks, such as image-text pairs.
   - Example:
     ```python
     from datasets import load_dataset
     dataset = load_dataset("derek-thomas/ScienceQA")
     ```

2. **Preprocessing**:
   - Tokenize text and preprocess images.
   - Example for tokenization:
     ```python
     model_id = 'microsoft/Florence-2-large'
     model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map='cuda')
     processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
     ```

3. **Data Collation**:
   - Ensure proper batching of image-text data.
   - Example:
     ```python
     def collate_fn(batch):
        questions, answers, images = zip(*batch)
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True)
        return inputs, answers
     ```

## Fine-Tuning

1. **Model Initialization**:
   - Load the pre-trained model:
     ```python
     from transformers import AutoModel
     model = AutoModel.from_pretrained("microsoft/florence-2-large", trust_remote_code=True, device_map='cuda')
     ```
2. **Enable Gradient Checkpointing**
    - Reduce the memory usage during neural network training:
    ```python
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    ```

3. **Training Configuration**:
   - Define training arguments and optimizers:
     ```python
        epochs = 2
        num_training_steps = epochs * len(train_loader)
        optimizer = AdamW(params=model.parameters(), lr=1e-5, optim_bits=8, is_paged=True)
        lr_sheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
     ```
4. **Gradient Accumulation**:
   - Gradient accumulation for effective batch size.
     ```python
     gradient_accumulation_steps = 8
     if i % gradient_accumulation_steps == 0:
        optimizer.step()
        lr_sheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()
     ```
5. **Training Loop**:
   - Use frameworks like `Trainer` or custom loops for fine-tuning.
     ```python
     for epoch in range(epochs):
        model.train()
        train_loss = 0
        i = -1
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            i += 1
            input_ids = inputs['input_ids'].to('cuda')
            pixel_values = inputs['pixel_values'].to('cuda')
            labels = processor.tokenizer(text=answers, return_tensors='pt', padding=True, return_token_type_ids=False)['input_ids'].to('cuda')
            output = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = output.loss/gradient_accumulation_steps
            loss.backward()
            if i % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_sheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            average_training_loss = train_loss / len(train_loader)
            print(f"Average training loss: {average_training_loss}")
     ```

## Usage

The fine-tuned model can be used for:

- Image captioning.
- Visual question answering.
- Image-text retrieval.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
