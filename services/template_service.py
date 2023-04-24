from typing import List
from langchain import PromptTemplate


class TemplateService:
    def create_template(self, template: str, input_variables: List[str]) -> PromptTemplate:
        return PromptTemplate(template=template, input_variables=input_variables)
