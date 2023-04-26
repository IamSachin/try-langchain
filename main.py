from services.template_service import TemplateService
from services.chain_service import ChainService
from models.openai import OpenAI

template = """ 
Question: {question}
Answer: 
"""
promptTemplate = TemplateService().create_template(template)
chain = ChainService(OpenAI()).get_chain(promptTemplate)

while True:
    question = input("Ask me anything!\n")

    if question.lower() == "quit":
        break

    print(chain.run(question))
    print('\n')
