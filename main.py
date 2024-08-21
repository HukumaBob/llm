import os
from llama_cpp import Llama

# Загрузка модели
model_path = "./llm/zephyr-7b-beta.Q4_0.gguf"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Файл модели не найден: {model_path}")
# GLOBAL VARIABLES
CONTEXT_SIZE = 512


# LOAD THE MODEL
zephyr_model = Llama(model_path=model_path,
                    n_ctx=CONTEXT_SIZE)

def generate_text_from_prompt(
        user_prompt,
        max_tokens = 100,
        temperature = 0.9,
        top_p = 0.1,
        echo = True,
        stop = ["Q", "\n"]
        ):
    # Define the parameters
    model_output = zephyr_model(
        user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        # stop=stop,
        )
    return model_output

if __name__ == "__main__":
    my_prompt = "Что любят собаки?"
    zephyr_model_response = generate_text_from_prompt(my_prompt)
    print('Ответ: ', zephyr_model_response)