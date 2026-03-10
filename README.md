# ⬡ NEXUS AI
### Next Generation Unified System

![Python](https://img.shields.io/badge/Python-3.10+-red?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-red?style=flat-square&logo=flask)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-red?style=flat-square)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-red?style=flat-square&logo=mongodb)
![Render](https://img.shields.io/badge/Deployed-Render-red?style=flat-square)

<br/>

> *Sharp. Composed. Always online.*

<br/>

</div>

---

## What I Built

NEXUS is a fully deployed personal AI assistant that I built from scratch — my own version of a Jarvis-style AI that lives in the browser.

It has a real brain, real memory, and can learn new things on its own. Unlike using ChatGPT or any other existing AI, NEXUS is completely mine — trained on my own data, designed with my own UI, remembers who I am, and gets smarter every time I talk to it.

It runs live on the internet and works on both desktop and mobile.

**Live at → [nexus-ai-vs58.onrender.com](https://nexus-ai-vs58.onrender.com)**

---

## What Makes It Different

Most chatbots forget everything the moment you close the tab. NEXUS doesn't.

It remembers your name, your preferences, your goals, and any instructions you give it — permanently. It stores everything in a cloud database so even if the server restarts, it still knows who you are when you come back.

It also doesn't just answer from stored knowledge. If you ask it something it doesn't know — like today's news, a recent sports result, or a current price — it searches the web, learns the answer, saves it, and responds as if it always knew it. No more "I don't have real-time information."

---

## How Each Feature Works

### 💬 AI Chat
Every message goes through a full pipeline before NEXUS responds:
1. Your message is searched against the RAG knowledge base
2. Previously learned knowledge is checked
3. NEXUS decides if a web search is needed
4. Memory, context, and knowledge are combined into a system prompt
5. Everything is sent to LLaMA 3.3 70B via the Groq API
6. The response is extracted, memory is updated, and the reply is returned

### 🧠 Persistent Memory
After every message, NEXUS scans what you said using keyword detection and saves anything worth remembering to MongoDB permanently:

- **Name** — picked up from phrases like "my name is" or "call me"
- **Facts** — picked up from "I am", "I work", "I love", "I live"
- **Instructions** — picked up from "always", "never", "don't", "from now on"
- **Preferences** — picked up from "respond in", "I prefer", "be more"
- **Goals** — picked up from "my goal is", "I want to", "I plan to"

All of this is loaded back every time you open the site. NEXUS already knows you before you say a word.

### 🌐 Web Search + Learning
When NEXUS detects a question that needs current information — news, results, prices, recent events — it does the following automatically:

1. Searches the web using Groq's built-in search tool
2. Reads and digests the raw results into 3–5 clean sentences
3. Saves the topic and knowledge to MongoDB permanently
4. Answers in its own voice as if it always knew the answer
5. Next time the same topic comes up — no search needed, already learned

A ⚡ LEARNED badge appears on the response whenever NEXUS learns something new.

### 📚 RAG Knowledge Base
NEXUS is trained on 12 custom dataset files I created — conversations covering topics like coding, space, motivation, philosophy, and more. At startup these are loaded and split into 347 chunks. Every message is matched against these chunks using TF-IDF keyword scoring, and the most relevant ones are injected into the system prompt to shape how NEXUS thinks and responds.

### ⏰ Date & Time Awareness
The current date and time is injected into every single system prompt automatically. NEXUS always knows exactly what time and day it is without needing to search for it.

### 🎨 Jarvis-Style UI
The interface is built with pure HTML, CSS, and JavaScript — no frameworks. It features a red HUD aesthetic with a boot sequence, animated hologram ring, particle network background, corner decorations, side system meters, and chat bubbles with timestamps. It was designed to feel like talking to a real AI system, not a webpage.

### 📱 Mobile Support
The UI fully adapts to mobile screens — side panels collapse, cards reflow into a 2-column grid, input font size prevents iOS zoom, and panels go full width. It works on both Android and iPhone browsers and can be added to the home screen to feel like a native app.

### 🗄️ MongoDB Persistent Storage
Memory and learned knowledge are stored in MongoDB Atlas — a free cloud database. This means data survives Render's free-tier restarts completely. If the server goes down and comes back, NEXUS reconnects to MongoDB and loads everything back instantly. If MongoDB is unavailable, it automatically falls back to local file storage.

</div>