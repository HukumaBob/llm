from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "llm/llama-7b.ggmlv3.q4_0.bin"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    user_input = input("Введите ваш запрос: ")
    response = generate_response(user_input)
    print("Ответ модели:", response)
