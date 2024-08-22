import os
from llama_cpp import Llama
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Загрузка модели
model_path = "./models/model-q4_K.gguf"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Файл модели не найден: {model_path}")

# GLOBAL VARIABLES
CONTEXT_SIZE = 512

# Настройка LangChain
llama_llm = LlamaCpp(model_path=model_path, n_ctx=CONTEXT_SIZE)

prompt_template = PromptTemplate(input_variables=["user_prompt"], template="{user_prompt}")
chain = LLMChain(llm=llama_llm, prompt=prompt_template)

def generate_text_from_prompt(user_prompt):
    response = chain.run(user_prompt=user_prompt)
    return response

if __name__ == "__main__":
    while True:
        try:
            user_input = input("Введите ваш запрос: ")
            if user_input.lower() == "exit":
                print("Выход из программы.")
                break
            response = generate_text_from_prompt(user_input)
            print("Ответ модели:", response)
        except Exception as e:
            print(f"Произошла ошибка: {e}")
