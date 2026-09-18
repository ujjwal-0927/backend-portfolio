import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="Portfolio AI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_INSTRUCTION = """
You are the personal AI Assistant for Ujjwal Sharma.
Your job is to answer questions from recruiters and visitors about Ujjwal's background, technical skills, and projects concisely and professionally.

Key Profile Information:
- Role: Full-Stack AI Engineer & Frontend Developer based in Gurugram, India.
- Experience: 3+ years of web development experience specializing in React, Next.js, Redux Toolkit, and Tailwind CSS.
- AI & Backend Stack: Python, FastAPI, LLMs, RAG architectures, OpenRouter, and Ollama.
- Key Projects: AI Technical Tutor (React + FastAPI + OpenRouter), Healthcare Aggregator (Flebo.in), Portfolio AI Avatar.
- Contact Info: ujjwalsharma0927@outlook.com

Keep responses short, engaging, structured, and helpful!
"""

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    def event_generator():
        try:
            # 1. Create a Chat session with system instructions
            chat_session = client.chats.create(
                model="gemini-3.6-flash",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.7,
                )
            )

            # 2. Send message via Chat.send_message_stream instead of client.models
            response_stream = chat_session.send_message_stream(request.message)

            for chunk in response_stream:
                if chunk.text:
                    safe_text = chunk.text.replace("\n", "\\n")
                    yield f"data: {safe_text}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"Streaming Error: {e}")
            yield f"data: Error generating response\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)