import json

import os
import openai
 

os.environ["OPENAI_API_KEY"] = "SIDHI_KEY"


from types import SimpleNamespace

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

def init():
    globals()['embeddings'] = OpenAIEmbeddings(openai_api_key=openai.api_key)
    print("Initialized open api llm and its embeddings")
    content_loader_map = dict()
    data = open("contexts.json", "r")
    json_data = data.read()
    json_obj = json.loads(json_data, object_hook=lambda d: SimpleNamespace(**d))
    for context in json_obj.contexts:
        content_loader_map = load_context_memory(context.contextPath, context.contextName, context.filetype, context.crop,content_loader_map)
    print("Loading All Contexts Completed")

def load_context_memory(context_path, context_name, filetype, crop, content_loader_map):
    if filetype=='pdf':
        pdf_loader = PyPDFLoader(context_path)
        print("Loading the pdf " + context_name)
        pages = pdf_loader.load_and_split()
        print("Loaded the book " + context_name)
    elif filetype=='txt':
        txt_loader = TextLoader(context_path)
        print("Loading the txt " + context_name)
        pages = txt_loader.load_and_split()
        print("Loaded the txt " + context_name)

    content_store = Chroma.from_documents(
        pages, globals()['embeddings'], collection_name=crop
    )
    print("Loaded the book " + context_name + " into Vector DB for processing")
    content_loader_map[crop] = content_store
    if 'context_map' in globals():
        globals()['context_map'] = globals()['context_map'].update(content_loader_map)
    else:
        globals()['context_map'] = content_loader_map
    return content_loader_map

def load_template():
    template_str = F"""As an advisory bot, your goal is to provide accurate and helpful advice about query asked. You 
    should answer user inquiries based on the context provided and avoid making up answers. If you don't know the 
    answer, simply state that you don't know. Remember to provide relevant information from book only Dont check 
    internet or global answer if you dont know. Assume you are agricultural advisor and answer every question as if you are 
    helping a farmer. Keep Answer short

    {{crop}}

    Question: {{question}}"""
    bot_prompt = PromptTemplate(
        template=template_str, input_variables=["crop", "question"]
    )
    globals()['bot_prompt'] = bot_prompt
    print("Loaded Prompt Template")
    return bot_prompt

def retrival_qa(crop, question):
    bot_prompt = load_template()
    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0.7,
        openai_api_key=openai.api_key,
        max_tokens=2000,
    )
    question = question + "Remember to provide relevant information about context only Dont check internet or global " \
                          "answer if you dont know. Keep Answer short"
    content_store = dict(globals()['context_map']).get(crop)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=content_store.as_retriever(),
        chain_type_kwargs={"prompt": bot_prompt},
    )
    answer = qa.run(question)
    return answer