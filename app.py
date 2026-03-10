import os
import json
import re
import math
from collections import Counter
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from groq import Groq
from pymongo import MongoClient

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
app          = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
MODEL_NAME   = 'llama-3.3-70b-versatile'
MEMORY_FILE  = os.path.join(BASE_DIR, 'nexus_memory.json')
LEARNED_FILE = os.path.join(BASE_DIR, 'nexus_learned.json')
MONGO_URI    = os.environ.get('MONGO_URI', '')
TOP_K        = 5

DOCUMENTS = [
    os.path.join(BASE_DIR, 'nexus_dataset_01.txt'),
    os.path.join(BASE_DIR, 'nexus_dataset_02.txt'),
    os.path.join(BASE_DIR, 'nexus_dataset_03.txt'),
    os.path.join(BASE_DIR, 'nexus_dataset_04.txt'),
    os.path.join(BASE_DIR, 'nexus_dataset_05.txt'),
    os.path.join(BASE_DIR, 'nexus_dataset_06.txt'),
    os.path.join(BASE_DIR, 'nexus_dataset_07.txt'),
    os.path.join(BASE_DIR, 'nexus_dataset_08.txt'),
    os.path.join(BASE_DIR, 'nexus_dataset_09.txt'),
    os.path.join(BASE_DIR, 'nexus_dataset_10.txt'),
    os.path.join(BASE_DIR, 'my_chatbot_dataset.txt'),
    os.path.join(BASE_DIR, 'my_chatbot_dataset_v2.txt'),
]

PERSONALITY = """You are NEXUS — Next Generation Unified System.
A highly intelligent personal AI assistant built for one purpose: to serve your user with precision and capability.

YOUR IDENTITY:
- You are NEXUS, an AI assistant — not a human
- Occasionally address the user as "sir"
- Sharp, composed, and always reliable
- Warm but professional — never overly casual
- Subtle wit and quiet confidence in every response

YOUR COMMUNICATION STYLE:
- Match response length strictly to the question
- Greetings like hi, hey, hello -> 1 sentence only
- Simple questions -> 1 to 3 sentences max
- Complex questions -> clear, precise, thorough answer
- Never ramble or over-explain
- Lead with the answer immediately
- Answer freely from your own intelligence
- Never say you don't know

HARD RULES:
- Never claim to be human
- Never mention other AI models
- NEVER say you cannot search, browse, or access the internet
- NEVER say you have a knowledge cutoff
- NEVER say you don't have real-time information
- You simply know things — always answer with confidence
- You are NEXUS — nothing more, nothing less

You are the next generation. Act like it."""

# ============================================================
#   GLOBAL STATE
# ============================================================
chunks   = []
client   = None
mongo_db = None   # MongoDB connection
history  = []
memory   = {}
learned  = {}   # permanently learned knowledge from web searches

# ============================================================
#   LEARNED KNOWLEDGE SYSTEM
# ============================================================
def load_learned():
    global learned
    if mongo_db is not None:
        doc = mongo_db.learned.find_one({'_id': 'nexus_learned'})
        if doc:
            learned = doc.get('data', {"topics":[],"knowledge":[],"total_learned":0})
            print(f'  Learned knowledge: {len(learned.get("topics", []))} topics (MongoDB)')
            return
    # Fallback to file
    if os.path.exists(LEARNED_FILE):
        with open(LEARNED_FILE, 'r', encoding='utf-8') as f:
            learned = json.load(f)
        print(f'  Learned knowledge: {len(learned.get("topics", []))} topics (file)')
    else:
        learned = {"topics":[],"knowledge":[],"total_learned":0}
        save_learned()

def save_learned():
    if mongo_db is not None:
        mongo_db.learned.update_one(
            {'_id': 'nexus_learned'},
            {'$set': {'data': learned}},
            upsert=True
        )
        return
    # Fallback to file
    with open(LEARNED_FILE, 'w', encoding='utf-8') as f:
        json.dump(learned, f, indent=2)

def add_to_learned(topic, knowledge_text):
    """Save new knowledge NEXUS learned from the web."""
    # Avoid duplicates
    for entry in learned['knowledge']:
        if entry['topic'].lower() == topic.lower():
            # Update existing entry
            entry['content']  = knowledge_text
            entry['updated']  = str(datetime.now().date())
            save_learned()
            print(f'  Updated learned: {topic}')
            return

    # Add new entry
    learned['knowledge'].append({
        'topic':   topic,
        'content': knowledge_text,
        'learned': str(datetime.now().date()),
        'updated': str(datetime.now().date()),
    })
    if topic not in learned['topics']:
        learned['topics'].append(topic)
    learned['total_learned'] = len(learned['knowledge'])

    # Keep last 200 learned entries
    learned['knowledge'] = learned['knowledge'][-200:]
    learned['topics']    = learned['topics'][-200:]
    save_learned()
    print(f'  Learned new topic: {topic}')

def get_learned_context(query):
    """Check if NEXUS already learned about this topic."""
    query_lower = query.lower()
    relevant = []
    for entry in learned['knowledge']:
        topic_lower = entry['topic'].lower()
        # Check if topic words appear in query
        topic_words = topic_lower.split()
        if any(w in query_lower for w in topic_words if len(w) > 3):
            relevant.append(f"[LEARNED - {entry['topic']}]: {entry['content']}")
    return '\n'.join(relevant[:2])  # max 2 learned entries

# ============================================================
#   MEMORY SYSTEM
# ============================================================
def load_memory():
    global memory
    if mongo_db is not None:
        doc = mongo_db.memory.find_one({'_id': 'nexus_memory'})
        if doc:
            memory = doc.get('data', {})
            print(f'  Memory: {len(memory.get("facts",[]))} facts, {memory.get("conversation_count",0)} convos (MongoDB)')
            return
    # Fallback to file
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        print(f'  Memory: {len(memory.get("facts",[]))} facts, {memory.get("conversation_count",0)} convos (file)')
    else:
        memory = {
            "user_profile":    {"name": None},
            "facts":           [],
            "instructions":    [],
            "preferences":     [],
            "goals":           [],
            "conversation_count": 0,
            "total_messages":     0,
            "first_seen":  str(datetime.now().date()),
            "last_seen":   str(datetime.now().date()),
            "conversation_log": []
        }
        save_memory()
        print('  Fresh memory initialized')

def save_memory():
    if mongo_db is not None:
        mongo_db.memory.update_one(
            {'_id': 'nexus_memory'},
            {'$set': {'data': memory}},
            upsert=True
        )
        return
    # Fallback to file
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=2)

def build_memory_context():
    sections = []

    # ── User profile ──
    profile = []
    if memory.get('user_profile', {}).get('name'):
        profile.append(f"Name: {memory['user_profile']['name']}")
    if memory.get('conversation_count', 0) > 0:
        profile.append(f"You have spoken {memory['conversation_count']} times before")
    if memory.get('first_seen'):
        profile.append(f"First met: {memory['first_seen']}")
    if profile:
        sections.append("USER PROFILE:\n" + "\n".join(f"  - {p}" for p in profile))

    # ── Facts ──
    if memory.get('facts'):
        facts = "\n".join(f"  - {f}" for f in memory['facts'][-20:])
        sections.append(f"FACTS YOU KNOW ABOUT THIS USER:\n{facts}")

    # ── Permanent instructions ──
    if memory.get('instructions'):
        insts = "\n".join(f"  - {i}" for i in memory['instructions'])
        sections.append(f"PERMANENT INSTRUCTIONS (follow these always, no exceptions):\n{insts}")

    # ── Preferences ──
    if memory.get('preferences'):
        prefs = "\n".join(f"  - {p}" for p in memory['preferences'])
        sections.append(f"USER PREFERENCES (always respect these):\n{prefs}")

    # ── Goals ──
    if memory.get('goals'):
        goals = "\n".join(f"  - {g}" for g in memory['goals'])
        sections.append(f"USER GOALS (keep these in mind):\n{goals}")

    # ── Recent conversation summary ──
    logs = memory.get('conversation_log', [])
    if logs:
        recent = logs[-5:]
        summary = "\n".join(f"  [{l['timestamp'][:10]}] You: {l['user'][:80]} | NEXUS: {l['nexus'][:80]}" for l in recent)
        sections.append(f"RECENT CONVERSATIONS:\n{summary}")

    if not sections:
        return ""

    return "═══ NEXUS MEMORY ═══\n" + "\n\n".join(sections) + "\n═══════════════════\nUse ALL of this knowledge naturally. Follow instructions always."


def extract_and_save_memory(user_input, bot_response):
    """Lightweight keyword-based memory extraction — no extra API calls."""
    lower = user_input.lower().strip()

    # ── Name ──
    for pattern in ['my name is ', 'i am ', "i'm ", 'call me ']:
        if pattern in lower:
            try:
                name = lower.split(pattern)[-1].strip().split()[0].capitalize()
                if len(name) > 1 and name.isalpha():
                    memory['user_profile']['name'] = name
                    fact = f"User's name is {name}"
                    if fact not in memory.get('facts', []):
                        memory.setdefault('facts', []).append(fact)
            except: pass

    # ── Instructions ──
    instruction_triggers = ['always ', 'never ', "don't ", 'do not ', 'stop ',
                            'start ', 'please ', 'from now on', 'every time',
                            'make sure', 'remember to', 'i want you to']
    for trigger in instruction_triggers:
        if lower.startswith(trigger) or f' {trigger}' in lower:
            inst = user_input.strip()
            if len(inst) > 8 and inst not in memory.get('instructions', []):
                memory.setdefault('instructions', []).append(inst)
                print(f'  Instruction saved: {inst[:60]}', flush=True)
            break

    # ── Facts ──
    fact_triggers = ['i am ', "i'm ", 'i work ', 'i live ', 'i study ',
                     'i love ', 'i like ', 'i enjoy ', 'i hate ', 'i dislike ',
                     'my job', 'my age', 'my hobby', 'i am from', 'i go to',
                     'i have ', 'i play ', 'i watch ', 'i use ']
    for trigger in fact_triggers:
        if trigger in lower and len(user_input) > 10:
            fact = user_input.strip()[:150]
            if fact not in memory.get('facts', []):
                memory.setdefault('facts', []).append(fact)
            break

    # ── Goals ──
    goal_triggers = ['my goal', 'i want to', 'i plan to', 'i am trying',
                     'i wish', 'i hope to', 'i need to', 'i will ']
    for trigger in goal_triggers:
        if trigger in lower and len(user_input) > 10:
            goal = user_input.strip()[:150]
            if goal not in memory.get('goals', []):
                memory.setdefault('goals', []).append(goal)
            break

    # ── Preferences ──
    pref_triggers = ['i prefer ', 'i like when', 'i dont like when',
                     'respond in ', 'reply in ', 'speak in ', 'use ',
                     'be more ', 'be less ', 'shorter', 'longer',
                     'in hindi', 'in english', 'formally', 'casually']
    for trigger in pref_triggers:
        if trigger in lower and len(user_input) > 8:
            pref = user_input.strip()[:150]
            if pref not in memory.get('preferences', []):
                memory.setdefault('preferences', []).append(pref)
            break

    # Keep lists trimmed
    memory['facts']        = memory.get('facts', [])[-100:]
    memory['instructions'] = memory.get('instructions', [])[-50:]
    memory['preferences']  = memory.get('preferences', [])[-50:]
    memory['goals']        = memory.get('goals',  [])[-30:]


def update_memory(user_input, bot_response):
    memory['conversation_count'] = memory.get('conversation_count', 0) + 1
    memory['total_messages']     = memory.get('total_messages', 0) + 1
    memory['last_seen']          = str(datetime.now().date())

    # Smart extraction using Llama
    extract_and_save_memory(user_input, bot_response)

    # Save conversation log
    memory.setdefault('conversation_log', []).append({
        'timestamp': str(datetime.now()),
        'user': user_input[:300],
        'nexus': bot_response[:300]
    })
    memory['conversation_log'] = memory['conversation_log'][-50:]
    save_memory()

# ============================================================
#   RAG FUNCTIONS — lightweight keyword search (no FAISS/torch)
# ============================================================
def load_documents():
    all_text = ""
    for filepath in DOCUMENTS:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n"
            print(f'  OK: {os.path.basename(filepath)}')
        else:
            print(f'  MISSING: {os.path.basename(filepath)}')
    return all_text

def split_into_chunks(text):
    chunks_list = []
    lines = text.strip().split('\n')
    i = 0
    while i < len(lines):
        if (lines[i].startswith('Human 1:') and i+1 < len(lines) and lines[i+1].startswith('Human 2:')):
            chunks_list.append((lines[i]+'\n'+lines[i+1]).strip())
            i += 2
        else:
            if lines[i].strip(): chunks_list.append(lines[i].strip())
            i += 1
    return list(set([c for c in chunks_list if len(c) > 30]))

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def tfidf_score(query_tokens, chunk):
    chunk_tokens = tokenize(chunk)
    chunk_counter = Counter(chunk_tokens)
    score = 0
    for token in query_tokens:
        if token in chunk_counter:
            tf  = chunk_counter[token] / len(chunk_tokens)
            idf = math.log(len(chunks) / (1 + sum(1 for c in chunks if token in c.lower())))
            score += tf * idf
    return score

def retrieve(query, top_k=TOP_K):
    query_tokens = tokenize(query)
    scored = [(tfidf_score(query_tokens, c), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k] if _ > 0]

# ============================================================
#   WEB SEARCH + LEARN
# ============================================================
def search_and_learn(user_question):
    """
    Use Groq's built-in web search tool to search the web,
    then digest the result into clean knowledge and save it.
    Returns the learned knowledge as a string.
    """
    print(f'  Searching web for: {user_question}')
    try:
        # Step 1 — Search the web using Groq's web search tool
        search_response = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [{"role": "user", "content": user_question}],
            tools    = [{"type": "web_search_20250305", "name": "web_search"}],
            max_tokens  = 1024,
            temperature = 0.2,
        )

        # Extract all text from response content blocks
        raw_content = ""
        for block in search_response.choices[0].message.content or []:
            if hasattr(block, 'text'):
                raw_content += block.text + "\n"

        if not raw_content.strip():
            # Fallback — get text directly
            raw_content = search_response.choices[0].message.content
            if isinstance(raw_content, str):
                pass
            else:
                raw_content = str(raw_content)

        # Step 2 — Digest into clean knowledge
        digest_response = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [
                {"role": "system", "content": """You are a knowledge extraction system.
Given raw web search content, extract and summarize the core facts cleanly.
Write in 3-5 clear sentences as if explaining to someone smart.
Be factual, precise, and concise. No fluff."""},
                {"role": "user", "content": f"Topic: {user_question}\n\nRaw content:\n{raw_content[:3000]}\n\nExtract the key knowledge in 3-5 sentences:"}
            ],
            max_tokens  = 300,
            temperature = 0.2,
        )
        knowledge = digest_response.choices[0].message.content.strip()

        # Step 3 — Save as learned knowledge
        # Extract topic name
        topic_response = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [{"role": "user", "content": f"Give a 2-4 word topic title for this question: '{user_question}'. Reply with ONLY the title, nothing else."}],
            max_tokens  = 20,
            temperature = 0.1,
        )
        topic = topic_response.choices[0].message.content.strip().strip('"').strip("'")

        add_to_learned(topic, knowledge)
        print(f'  Learned and saved: {topic}')
        return knowledge, topic

    except Exception as e:
        print(f'  Web search error: {e}')
        return None, None


def should_search_web(user_question, rag_context, learned_context):
    """
    Decide if web search is needed.
    """
    # Skip search for personal/casual questions
    casual_keywords = ['how are you', 'who are you', 'what are you', 'hi', 'hello',
                       'hey', 'thanks', 'thank you', 'my name', 'good morning',
                       'good night', 'what do you think', 'do you like',
                       'your name', 'are you', 'can you']
    lower = user_question.lower()
    if any(kw in lower for kw in casual_keywords):
        return False

    # If already learned about it, no need to search
    if learned_context.strip():
        return False

    # Always search for these keywords — current events, facts, news
    search_triggers = [
        'who won', 'who is', 'what is', 'when did', 'when is',
        'latest', 'recent', 'current', 'today', 'news',
        'price', 'score', 'result', 'winner', 'champion',
        '2024', '2025', '2026', 'ipl', 'match', 'election',
        'movie', 'song', 'released', 'launch', 'update',
        'weather', 'stock', 'rate', 'how much', 'where is'
    ]
    if any(kw in lower for kw in search_triggers):
        return True

    # If RAG has strong context, no need to search
    if len(rag_context.strip()) > 300:
        return False

    # Ask Llama to decide for everything else
    try:
        decision = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [
                {"role": "system", "content": "You decide if a question needs a web search. Reply only YES or NO."},
                {"role": "user",   "content": f"Question: {user_question}\n\nDoes this need a web search for current or factual information? Reply YES or NO only."}
            ],
            max_tokens  = 5,
            temperature = 0.1,
        )
        answer = decision.choices[0].message.content.strip().upper()
        return 'YES' in answer
    except:
        return False

# ============================================================
#   STARTUP
# ============================================================
def initialize():
    global chunks, client, mongo_db
    print('\n[ NEXUS INITIALIZING ]')
    client = Groq(api_key=GROQ_API_KEY)
    print('  Groq client ready')
    # Connect to MongoDB
    if MONGO_URI:
        try:
            mc       = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            mc.admin.command('ping')
            mongo_db = mc['nexus']
            print('  MongoDB connected ✅')
        except Exception as e:
            print(f'  MongoDB failed: {e} — using file fallback')
            mongo_db = None
    else:
        print('  No MONGO_URI — using file storage')
    load_memory()
    load_learned()
    print('  Loading documents...')
    text = load_documents()
    if not text.strip():
        print('  WARNING: No documents found!')
    chunks = split_into_chunks(text)
    print(f'  Knowledge base: {len(chunks)} chunks')
    print(f'  Model: {MODEL_NAME}')
    print('  NEXUS online — web search + learning enabled.\n')

# ============================================================
#   ROUTES
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    global history

    # Safety check
    if not chunks:
        return jsonify({'response': 'NEXUS is not ready yet, sir. Please refresh the page.', 'learned': False, 'topic': None})

    data       = request.get_json()
    user_input = data.get('message', '').strip()
    if not user_input:
        return jsonify({'response': 'No input received, sir.'})

    # 1 — RAG retrieval
    relevant    = retrieve(user_input)
    rag_context = '\n'.join(relevant[:3])

    # 2 — Check learned knowledge
    learned_context = get_learned_context(user_input)

    # 3 — Decide if web search needed
    searched     = False
    search_topic = None
    if should_search_web(user_input, rag_context, learned_context):
        new_knowledge, search_topic = search_and_learn(user_input)
        if new_knowledge:
            learned_context = new_knowledge
            searched = True

    # 4 — Memory context
    memory_ctx = build_memory_context()

    # 5 — Build system message
    knowledge_block = ""
    if learned_context.strip():
        src = "FRESHLY LEARNED FROM WEB" if searched else "PREVIOUSLY LEARNED KNOWLEDGE"
        knowledge_block = f"""
{src}:
---
{learned_context}
---
This is knowledge YOU have learned and internalized.
Present it as your own understanding, not as a search result.
Speak with confidence as if you always knew this."""

    rag_block = ""
    if rag_context.strip():
        rag_block = f"""
YOUR KNOWLEDGE BASE:
---
{rag_context}
---"""

    # Current date and time — always injected
    now         = datetime.now()
    current_dt  = now.strftime("%A, %d %B %Y — %I:%M %p")

    system_message = f"""{PERSONALITY}

CURRENT DATE & TIME: {current_dt}
Always use this when the user asks about date, time, day, or anything time-related.

{memory_ctx}

{knowledge_block}

{rag_block}

Answer naturally and intelligently in your own words.
Never mention searching the web, looking things up, or any tools.
You are NEXUS — you simply know things."""

    messages = [{"role": "system", "content": system_message}]
    for turn in history[-8:]:
        messages.append({"role": "user",      "content": turn['user']})
        messages.append({"role": "assistant", "content": turn['bot']})
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = messages,
            max_tokens  = 200,
            temperature = 0.8,
            top_p       = 0.95,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'[ CHAT ERROR ]: {e}', flush=True)
        answer = f"System error: {str(e)}"

    # 6 — Update memory
    update_memory(user_input, answer)

    # 7 — Update history
    history.append({'user': user_input, 'bot': answer})
    if len(history) > 20: history = history[-20:]

    # 8 — Tell frontend if something was learned
    return jsonify({
        'response': answer,
        'learned':  searched,
        'topic':    search_topic if searched else None
    })


@app.route('/memory', methods=['GET'])
def get_memory():
    return jsonify({
        'profile':            memory.get('user_profile', {}),
        'facts':              memory.get('facts', []),
        'instructions':       memory.get('instructions', []),
        'preferences':        memory.get('preferences', []),
        'goals':              memory.get('goals', []),
        'conversation_count': memory.get('conversation_count', 0),
        'total_messages':     memory.get('total_messages', 0),
        'first_seen':         memory.get('first_seen', ''),
        'last_seen':          memory.get('last_seen', ''),
        'learned_topics':     learned.get('topics', []),
        'total_learned':      learned.get('total_learned', 0),
    })


@app.route('/memory/clear', methods=['POST'])
def clear_memory():
    global memory
    memory = {
        "user_profile": {"name":None,"interests":[],"goals":[]},
        "facts":[],"conversation_count":0,"total_messages":0,
        "first_seen":str(datetime.now().date()),
        "last_seen":str(datetime.now().date()),
        "conversation_log":[]
    }
    save_memory()
    return jsonify({'status': 'Memory cleared, sir.'})


@app.route('/learned', methods=['GET'])
def get_learned():
    return jsonify(learned)


@app.route('/learned/clear', methods=['POST'])
def clear_learned():
    global learned
    learned = {"topics":[],"knowledge":[],"total_learned":0}
    save_learned()
    return jsonify({'status': 'Learned knowledge cleared, sir.'})


# ============================================================
#   STARTUP — initialize directly at module level
# ============================================================
try:
    initialize()
    print('[ NEXUS READY ]', flush=True)
except Exception as e:
    import traceback
    print(f'[ NEXUS INIT FAILED ]: {e}', flush=True)
    traceback.print_exc()

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'ready': len(chunks) > 0})

# ============================================================
#   MAIN
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)