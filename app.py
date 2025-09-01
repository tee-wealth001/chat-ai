from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Gemma
model_name = "gemma-3-270m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# FastAPI app
app = FastAPI(title="LLM Answering")


class LLMQuery(BaseModel):
    question: str
    context: list
    history: list
    max_new_tokens: int = 200


@app.post("/answer")
def answer(query: LLMQuery):
    context_text = "\n\n".join(
        [f"{c['title']} ({c['url']}): {c['chunk']}" for c in query.context]
    )
    prompt = f"""You are a helpful assistant. 

        Answer the following question based on the provided context. 
        If the question is unclear or missing details, refer to the conversation history to clarify.

        Conversation history:
        {query.history}

        Context:
        {context_text}

        Question:
        {query.question}

        Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=query.max_new_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "").strip()
    return {"answer": answer}
