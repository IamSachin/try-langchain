from langchain import LLMChain, PromptTemplate
from models.model_provider import ModelProvider


class ChainService:
    def __init__(self, model: ModelProvider):
        self.model = model

    def get_chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            prompt=prompt,
            llm=self.model.llm()
        )
