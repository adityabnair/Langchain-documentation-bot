# LangChain Documentation Helper Bot
This is a LangChain based AI chatbot that aims to demonstrate how the usage of a vector database and document loaders to parse several documents into indices can be combined with a front-end chat interface like 
StreamLit to provide a seamless experience for anybody who wishes to enquire about LangChain and its coding practices.

Fun fact: If ChatGPT is asked as of March 2024, what LangChain is, it will be unable to answer (or in the case of GPT 4 look up the internet) since its information about this topic is limited.
![image](https://github.com/adityabnair/Langchain-documentation-bot/assets/64246274/517299e2-020b-4664-a191-1b4e1751c845)
This bot helps users who are new to LangChain and require an assistant of their own to help them through their LangChain coding journey.
The documentation folder for LangChain is slightly out of date as the latest one hosted at https://api.python.langchain.com/en/latest/langchain_api_reference.html seems to not allow direct downloads. However, this documentation data can be substituted as required. There is also an attached python script using BeautifulSoup library which can be used to donwload documents. 

The operation process is simple. The ingestion.py file gathers to locally downloaded documents which it then transfers to the relevant PineCone vector databases by splitting it into chunks, given the contextual 
embeddings. Next the backend packages uses the ConversationalRetreivalQA chain from the LangChain library to preserve the memory of a session state's chat history to allow the chatbot to remember the context of 
the conversation with the user. Finally the main.py file uses StreamLit library's inbuilt chat interface to allow the storage of three key components of the chat session state: the user's prompt history, the chat
answer history (to allow the display of previous messages) and the overall chat history (to combine the prompts and the generated response). The source URL's are also attached at the end of every response to display where the bot was able to get the information from, which further enables the user to verify the same. 

## Screenshots
![image](https://github.com/adityabnair/Langchain-documentation-bot/assets/64246274/006270cd-6424-4079-af0b-bad9496ee66a)
![image](https://github.com/adityabnair/Langchain-documentation-bot/assets/64246274/f29fd9da-d49b-4c72-a3d7-b13a1cd029df)
![image](https://github.com/adityabnair/Langchain-documentation-bot/assets/64246274/b96a8597-547b-4e5d-9b47-67e7f22defe3)
![image](https://github.com/adityabnair/Langchain-documentation-bot/assets/64246274/7983b347-23d6-4a0c-ae5d-6ac3fb00e014)



## Main Prerequisites

1. At least Python 3.10
2. Access to OpenAI's API credits for usage of gpt-3.5
3. Access to Pinecone's API and index creation

### Running

1. Use pipenv to install python libraries from requirements.txt (a virtual environemnt is always recommended)
2. Add environment variable in a .env file to hold Pinecone's and OpenAI's API keys as well as PYTHONPATH for the root directory of the project
3. The index at Pinecone should have 1536 dimensions and should be of the cosine type for vector matching
4. Consts.py should be updated with the name of the created Pinecone index
5. Run ingestion.py to load indexes to Pinecone
6. Run main.py with the ``` streamlit run <directory to main> ``` command


## Acknowledgments

Thanks to @emarco177 for the langchain development course
