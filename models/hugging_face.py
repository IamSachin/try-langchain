from langchain import HuggingFaceHub
from settings import FLAN_T5_XL, TEMP_VERY_COLD
from .model_provider import ModelProvider


class HuggingFace(ModelProvider):
    def __init__(self, model=FLAN_T5_XL):
        self.model = model

    def llm(self):
        return HuggingFaceHub(repo_id=self.model, model_kwargs={"temperature": TEMP_VERY_COLD})
