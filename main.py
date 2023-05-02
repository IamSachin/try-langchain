from services.template_service import TemplateService
from services.chain_service import ChainService
from models.openai import OpenAI
from langchain.chains import TransformChain, SequentialChain
from helpers.clean_input import clean_input
from helpers.count_token import count_token


# Sequential chaining
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


# Conversation chaining
conversation_buf = ChainService(OpenAI()).get_conversation_chain()


print('Select mode: ')
print('1. Sequential chain conversation')
print('2. Conversation with memory')
option = input("Enter option number: ")


while True:
    question = input("\nAsk me anything!\n")

    if question.lower() == "quit":
        break

    match option:
        case '1':
            count_token(sequential_chain, {'text': question})
        case '2':
            count_token(conversation_buf, question)
        case other:
            print('Invalid mode\n')
