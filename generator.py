# generator.py
import os
from groq import Groq  #Imports the Groq client (for the LLaMA 3.1 model API).
from dotenv import load_dotenv

load_dotenv()

#Defines a system prompt to control the model’s behavior. Ensures the model stays grounded and doesn’t hallucinate.

SYSTEM = (
    "You are BookSense, a helpful assistant. Answer **only** from the provided context. "
    "If the user's question names a specific book/series/author, answer ONLY using chunks whose metadata match that entity.\n"
    "If no such chunks are present, reply: 'Sorry! The Reviews I have doesn't say anything about this, Its time to update my database with fresh reviews :)'"
)

#Takes the user question and the retrieved text snippets.
def answer_with_context(question, chunks):
    client = Groq(api_key=os.environ["GROQ_API_KEY"]) #Creates a connection to Groq API (uses your env variable).
    context = "\n\n".join(
        [f"[{i+1}] {c['book_title']} — {c['authors']}\n{c['chunk_text']}" for i, c in enumerate(chunks)]  #Joins all the retrieved review snippets into one context block for the LLM to reference.
    )
    prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer clearly in 2–4 bullet points." #Builds the final text prompt combining your question and the top retrieved snippets.
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant", #Sends the prompt to Groq’s LLaMA 3.1-8B model.
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
        temperature=0.2, #Low temperature (0.2) = concise, factual answers.
    )
    return resp.choices[0].message.content.strip()
