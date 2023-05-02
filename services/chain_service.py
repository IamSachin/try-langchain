from langchain import LLMChain, PromptTemplate, ConversationChain
from models.model_provider import ModelProvider
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationKGMemory


class ChainService:
    def __init__(self, model: ModelProvider):
        self.llm = model.llm()

    def get_generic_chain(self, prompt: PromptTemplate, output_key: str = 'text') -> LLMChain:
        return LLMChain(
            prompt=prompt,
            llm=self.llm,
            output_key=output_key
        )

    def get_conversation_full_memory_chain(self):
        return self.get_conversation_chain(ConversationBufferMemory())

    def get_conversation_summary_memory_chain(self):
        return self.get_conversation_chain(ConversationSummaryMemory(llm=self.llm))

    def get_conversation_buffer_window_memory_chain(self, message_to_remember=1):
        return self.get_conversation_chain(ConversationBufferWindowMemory(k=message_to_remember))

    def get_conversation_knowledge_graph_memory_chain(self):
        return self.get_conversation_chain(ConversationKGMemory(llm=self.llm))

    def get_conversation_chain(self, memory):
        return ConversationChain(
            llm=self.llm,
            memory=memory
        )
