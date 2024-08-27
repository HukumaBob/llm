from functools import lru_cache
import os
import json
import logging
from llama_cpp import Llama
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import JSONLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document  # Изменение импорта

logging.basicConfig(level=logging.INFO)

# Загрузка модели
model_path = "./models/model-q4_K.gguf"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Файл модели не найден: {model_path}")

# GLOBAL VARIABLES
CONTEXT_SIZE = 512

# Настройка LangChain
llama_llm = LlamaCpp(model_path=model_path, n_ctx=CONTEXT_SIZE)

prompt_template = PromptTemplate(
    input_variables=["user_prompt", "language"],
    template="{user_prompt} на языке {language}"
)
chain = LLMChain(llm=llama_llm, prompt=prompt_template)

# Загрузка документов
def load_json_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                data = json.load(file)
                for entry in data:
                    # Объединение всех текстовых полей в один текст
                    combined_text = (
                        f"UIK Number: {entry.get('uik_number', '')}\n"
                        f"UIK Address: {entry.get('uik_address', '')}\n"
                        f"City/Town/Region: {entry.get('city_town_region', '')}\n"
                        f"Simplified Borders: {entry.get('simplified_borders', '')}\n"
                        f"UIK Borders: {entry.get('uik_borders', '')}"
                    )
                    document = Document(page_content=combined_text, metadata={"id": str(entry.get('uik_number', ''))})
                    documents.append(document)
    return documents

# Создание документов с уникальными идентификаторами
json_documents = load_json_documents("./documents")

# Создание векторного хранилища
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embeddings)

# Добавление документов в векторное хранилище
vector_store.add_documents(json_documents)

def search_documents(query):
    results = vector_store.similarity_search(query, k=5)
    return results

@lru_cache(maxsize=100)
def generate_text_from_prompt(user_prompt, language="ru"):
    try:
        search_results = search_documents(user_prompt)
        logging.info(f"Результаты поиска: {search_results}")
        
        # Используйте только первый результат для контекста, чтобы уменьшить количество токенов
        if search_results:
            context = search_results[0].page_content
        else:
            context = ""
        
        user_prompt += f" Контекст: {context}"
        
        total_tokens = llama_llm.tokenizer(user_prompt)["input_ids"]
        if len(total_tokens) > CONTEXT_SIZE:
            raise ValueError("Количество токенов превышает допустимый лимит контекста")

        response = chain.run(user_prompt=user_prompt, language=language)
        logging.info(f"Запрос: {user_prompt}, Ответ: {response}")
        return response
    except Exception as e:
        logging.error(f"Ошибка при генерации текста: {e}")
        return "Произошла ошибка при обработке вашего запроса."


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
