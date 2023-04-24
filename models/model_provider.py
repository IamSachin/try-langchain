from abc import ABCMeta, abstractmethod


class ModelProvider(metaclass=ABCMeta):
    @abstractmethod
    def llm(self):
        pass
