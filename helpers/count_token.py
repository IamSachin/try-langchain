from langchain.callbacks import get_openai_callback


def count_token(chain, input):
    with get_openai_callback() as cb:
        print(chain.run(input))
        print('\n')
        print(cb)
        print('\n')
