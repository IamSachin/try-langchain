from langchain import HuggingFaceHub
from settings import FLAN_T5_XL, TEMP_VERY_COLD
from .model_provider import ModelProvider


class HuggingFace(ModelProvider):
    def __init__(self, model=FLAN_T5_XL, temperature=TEMP_VERY_COLD):
        self.model = model
        self.temperature = temperature

    def llm(self):
        return HuggingFaceHub(repo_id=self.model, model_kwargs={"temperature": self.temperature})
