from fastapi import FastAPI
from llm_langchain import LLMLangchain
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok

port = 8000

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llmLangchain = LLMLangchain("llama")


print("server listening...")

@app.get("/api/{que}")
async def get_ans(que):
    print(que)
    ans, source = llmLangchain.get_answer(que)
    return { 
        "answer": ans,
        "source": source
    }
    # return {"answer" : "hello"}

@app.get("/")
async def home():
    return { "response": "welcome to llm api" }
# public_url = ngrok.connect("8000").public_url
# print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))


# llmLangchain.get_answer("How can I go through sacrifices and wealth in my life ?")
