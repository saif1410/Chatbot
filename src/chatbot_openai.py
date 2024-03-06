#!/usr/bin/env python
# coding: utf-8

#import the necessary libraries

import llama_index
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import openai


# set up environment and the keys 

openai.api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] ="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"

openai.api_key = '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'


# function to construct indexes for our given document

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    
    max_chunk_overlap=1
    
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data() 

    index = GPTVectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    #provide the path to save the json indexes created by llama-index
    index.storage_context.persist(persist_dir='/home/mirafra/chatbot/index/index.json')

    return index



construct_index("/home/mirafra/chatbot/pdf/newdocument.pdf") #provide the path of the pdf document for index creation


from llama_index import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir='/home/mirafra/chatbot/index/index.json')  
#provide the location of json index that's already created by our construct_index function

def chatbot(input_text):
    index = load_index_from_storage( storage_context )
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)  #input text is argument of the user
    return response.response


#gradio interface for interacting with the chat-bot

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-knowledge trained AI Chatbot")


iface.launch(share=True)  #launch the interface



