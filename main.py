from services.template_service import TemplateService
from services.chain_service import ChainService
from models.openai import OpenAI
from langchain.chains import TransformChain, SequentialChain
from helpers.clean_input import clean_input
from helpers.count_token import count_token


open_ai_chain = ChainService(OpenAI())

# Sequential chaining
clean_extra_spaces_chain = TransformChain(input_variables=['text'], output_variables=[
                                          'output_text'], transform=clean_input)

template = """
Question: {output_text}
Answer:
"""
promptTemplate = TemplateService().create_template(template)
basic_generic_chain = open_ai_chain.get_generic_chain(
    promptTemplate, output_key='final_output')
sequential_chain = SequentialChain(
    chains=[clean_extra_spaces_chain, basic_generic_chain],
    input_variables=['text'], output_variables=['final_output'])


# Memory chains
full_memory_chain = open_ai_chain.get_conversation_full_memory_chain()
summary_memory_chain = open_ai_chain.get_conversation_summary_chain()

# Menu
print('Select mode: ')
print('1. Sequential chain conversation - clean up and then simple q and a without memory')
print('2. Conversation with complete memory - more accurate, good for short conversations')
print('3. Conversation with summarization memory - less accurate, good for long conversations')
option = input("Enter option number: ")


while True:
    question = input("\nAsk me anything!\n")

    if question.lower() == "quit":
        break

    match option:
        case '1':
            count_token(sequential_chain, {'text': question})
        case '2':
            count_token(full_memory_chain, question)
        case '3':
            count_token(summary_memory_chain, question)
        case other:
            print('Invalid mode\n')
