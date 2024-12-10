# Projeto de Assistente Médico com LoRA e FAISS

Este projeto utiliza um modelo de linguagem adaptado com LoRA (Low-Rank Adaptation) e FAISS para criar um assistente médico que pode responder a perguntas com base em documentos contextuais.

## Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2. Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
    ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Fine-Tuning do Modelo

Para realizar o fine-tuning do modelo utilizando LoRA, siga os passos abaixo:

1. Carregue o modelo LoRA e o tokenizer:
    ```python
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model_medico_conversa_v2",  # Seu modelo treinado
        max_seq_length = 512,  # Comprimento máximo da sequência
        dtype = "float16",  # Tipo de dado
        load_in_4bit = True,  # Carregar em 4 bits
    )
    FastLanguageModel.for_inference(model)  # Habilita inferência 2x mais rápida
    ```

2. Defina a mensagem do usuário e gere a resposta:
    ```python
    messages = [
        {"role": "user", "content": "Qual é a importância das relações interpessoais na saúde mental?"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(input_ids, streamer=text_streamer, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    ```

## Contribuição

Sinta-se à vontade para abrir issues e pull requests para contribuir com melhorias para este projeto.

## Licença

Este projeto está licenciado sob os termos da licença MIT.
