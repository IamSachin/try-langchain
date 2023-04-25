from langchain.llms import OpenAI as OpenAILLM
from settings import TEXT_DAVINCI_003, TEMP_OPENAI_DEFAULT
from .model_provider import ModelProvider


class OpenAI(ModelProvider):
    def __init__(self, model=TEXT_DAVINCI_003, temperature=TEMP_OPENAI_DEFAULT):
        self.model = model
        self.temperature = temperature

    def llm(self):
        return OpenAILLM(model_name=self.model, temperature=self.temperature)
