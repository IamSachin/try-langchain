from langchain.llms import OpenAI as OpenAILLM
from settings import TEXT_DAVINCI_003
from .model_provider import ModelProvider


class OpenAI(ModelProvider):
    def __init__(self, model=TEXT_DAVINCI_003):
        self.model = model

    def llm(self):
        return OpenAILLM(model_name=self.model)
