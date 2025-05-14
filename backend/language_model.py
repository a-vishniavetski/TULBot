import os

from fastapi import HTTPException
from huggingface_hub import hf_hub_download, login
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import groq


def setup_local_language_model():
    """
    Returns (model, tokenizer, generation_config) for the language model.
    """
    hf_token = os.getenv('hf_token')
    if hf_token is None:
        raise ValueError("Hugging Face token not found in environment variables.")
    login(token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained("eryk-mazus/polka-1.1b-chat")
    model = AutoModelForCausalLM.from_pretrained("eryk-mazus/polka-1.1b-chat")

    # Set proper generation configuration
    generation_config = GenerationConfig(
        max_new_tokens=128,  # Controls the maximum length of the generation
        do_sample=True,      # Enable sampling
        temperature=0.65,     # Control randomness
        top_p=0.9,           # Nucleus sampling
        top_k=10,            # Top-k sampling
        repetition_penalty=1.4,  # Penalize repetition
        pad_token_id=tokenizer.eos_token_id  # Properly set pad token
    )   
    return tokenizer, model, generation_config

def get_answer_from_local(system_prompt, query: str, tokenizer, model) -> str:
    try:
        inputs = tokenizer(system_prompt, return_tensors="pt")

        print(f"Generating response for prompt...")

        outputs = model.generate(**inputs, generation_config=generation_config)
        answer =  tokenizer.decode(outputs[0], skip_special_tokens=True)

        if system_prompt in answer:
            answer = answer.split(system_prompt)[-1].strip()

        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

def get_asnwer_from_groq(system_prompt, query: str) -> str:
    try:
        client = groq.Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        answer = chat_completion.choices[0].message.content
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")