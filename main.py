from services.template_service import TemplateService
from services.chain_service import ChainService
from models.openai import OpenAI
from langchain.callbacks import get_openai_callback

template = """ 
Question: {question}
Answer: 
"""
promptTemplate = TemplateService().create_template(template)
chain = ChainService(OpenAI()).get_chain(promptTemplate)

while True:
    with get_openai_callback() as cb:
        question = input("Ask me anything!\n")

        if question.lower() == "quit":
            break

        print(chain.run(question))
        print('\n')
    print(cb)
    print('\n')
