---
title: "RAG: Why we need it and how to implement it"
date: 2024-06-20T14:49:39+08:00
draft: false # Set 'false' to publish
tableOfContents: true # Enable/disable Table of Contents
description: 'Sharing Some Tips I learned from my RAG application'
categories:
  - Tutorials
tags:
  - RAG
---

Recently I have been working on a Research Agent project. The project is essentially a multi-agent system. Through the system, an entire research pipeline, including coming up with research questions, methods, and experiments, can all be automated and powered by LLMs. The models need to be trained according to their designated tasks to maximize output quality, and a RAG(Retrieval-Augmented Generation) is mostly required for the agents to generate meaningful output. 

When implementing the RAG system, I encountered some difficulties when setting it up. Today I will share a simplified version of setting RAG up and hopefully this method could help you get RAG done.

## What is RAG and Why?

If you have played with one of the chatbots, you would probably have noticed that depending on your input prompt, the chatbot would generate significantly different answers. Interestingly, the information contained in your prompt, the length of context, and even the position of important details in your prompt, can greatly influence the performance of those LLMs. This ability of models is called in-context learning, and researchers have been studying prompt engineering to facilitate model performance.

RAG, in a sense, is a type of prompt engineering. Imagine you want to ask a sophisticated question, one that might require extra information to process. You *could* potentially include extra information in your prompt. But there are problems with hand-picking related sources. How do you determine if the document you include is relevant and helpful to the question you asked? The time and effort you need is just not worth it. 

RAG comes to the rescue under this circumstance. It maintains a large database that contains a large number of documents. For the question you ask, it uses your question to find relevant documents for you and combine those documents with your question as input to the language models. The magic behind it is fairly simple: RAG uses a similarity score to get the most relevant document with your question. There are plenty of other details that you could use to customize its behavior of course. 

## How to implement RAG
The easiest way to do it is probably through LangChain. Langchain is a framework that enables fast multi-agent system development. It has built-in support for RAG databases like Chromadb. 

### Imports and API Key
Before setting Chromadb up, remember to import essential packages and setup your API key. You can get an API key from the langchain official website.
```python
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings 
from langchain_community.vectorstores import Chroma

os.environ['LANGCHAIN_ENDPOINT']  = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "your_api_key"
```

### Chromadb Setup
Setup a Chromadb vector store locally is as easy as:
```python
def vectorstore_load(save_path: str = "./db", collection_name: str = "rag-chroma"):
    # Load vectorstore
    ef = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=save_path, collection_name = collection_name, embedding_function=ef)
    return vectorstore
```
It is called 'vector store' because all sentences are embedded as vectors in the database. "all-MiniLM-L6-v2" is the default model for the embeddings. You could change it to another model to improve performance, but that's unnecessary.

### Save Document to RAG
After setting it up, you could save documents to your RAG vector store like this:
```python
def save_to_vectorstore(document: str, save_path: str = "./db", collection_name: str = "rag-chroma"):
    vectorstore = vectorstore_load()
    vectorstore.add_documents([Document(document)])
```
**Notice** that save path and collection name has to be the same. Using a different path or name would create a separate database. 

### Document Verification
To verify the documents are stored correctly, you could check how many documents you have stored:
```python
vectorstore = vectorstore_load()
print(len(vectorstore.get()['documents']))
```

### Retrieve Relevant Documents
Now, you can use the following code to get relevant documents from your database
```python
retriever = vectorstore.as_retriever()
question = "how to save llm costs?"
docs = retriever.invoke(question)
```

## Conclusion
That's it! As easy as that. You now get the relevant documents and you could prompt them to your LLM together with your question. 