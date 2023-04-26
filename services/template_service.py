from typing import List
from langchain import PromptTemplate


class TemplateService:
    def create_template(self, template: str) -> PromptTemplate:
        return PromptTemplate.from_template(template=template)
