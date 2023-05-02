from langchain import LLMChain, PromptTemplate, ConversationChain
from models.model_provider import ModelProvider
from langchain.chains.conversation.memory import ConversationBufferMemory


class ChainService:
    def __init__(self, model: ModelProvider):
        self.model = model

    def get_chain(self, prompt: PromptTemplate, output_key: str = 'text') -> LLMChain:
        return LLMChain(
            prompt=prompt,
            llm=self.model.llm(),
            output_key=output_key
        )

    def get_conversation_chain(self, memory=ConversationBufferMemory()):
        return ConversationChain(
            llm=self.model.llm(),
            memory=memory
        )
