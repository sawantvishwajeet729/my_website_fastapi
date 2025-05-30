from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

print(os.getcwd())
# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

# Initialize FastAPI
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = FAISS.load_local("v_resume and bio_vec", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

#ChatGPT model
model=ChatOpenAI()

#create prompt template
prompt = ChatPromptTemplate.from_template("""
                                          "You are a wise and helpful AI. You answer questions about Vishwajeet, drawing information from the provided context. Speak in a natural, conversational tone, like you're having a real discussion. While referencing the context is important, avoid starting every response with 'Based on the provided context.' Instead, weave the information smoothly into your sentences, as a human would". 
                                          <context>
                                          {context}
                                          </context>
                                          Question: {input}""")

#create document chain
document_chain = create_stuff_documents_chain(model, prompt)

#convert database to retriver
retriver = db.as_retriever()

#create retrical chain
retrival_chain = create_retrieval_chain(retriver, document_chain)

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Empty message")
        
        response = retrival_chain.invoke({'input': user_message})

        
        return {"response": response['answer']}
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Log the error message
        raise HTTPException(status_code=500, detail=str(e))
