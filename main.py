from services.template_service import TemplateService
from services.chain_service import ChainService
from models.openai import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import TransformChain, SequentialChain
from helpers.clean_input import clean_input

# First chain that cleans up the input
clean_extra_spaces_chain = TransformChain(input_variables=['text'], output_variables=[
                                          'output_text'], transform=clean_input)

template = """
Question: {output_text}
Answer:
"""
promptTemplate = TemplateService().create_template(template)
basic_generic_chain = ChainService(OpenAI()).get_chain(
    promptTemplate, output_key='final_output')

sequential_chain = SequentialChain(
    chains=[clean_extra_spaces_chain, basic_generic_chain],
    input_variables=['text'], output_variables=['final_output'])


while True:
    with get_openai_callback() as cb:
        question = input("Ask me anything!\n")

        if question.lower() == "quit":
            break

        print(sequential_chain.run({'text': question}))
        print('\n')
    print(cb)
    print('\n')
