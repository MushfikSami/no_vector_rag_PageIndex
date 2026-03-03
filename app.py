import os
import json
import re
import time
import hashlib
import asyncio
from datetime import datetime
import gradio as gr
from openai import OpenAI
from curl_cffi.requests import AsyncSession
from selectolax.parser import HTMLParser
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. Configuration & Setup
# ==========================================
# Local LLM Config
LOCAL_API_BASE = "http://localhost:5000/v1"  # Or 5000 based on your vLLM setup
LOCAL_MODEL = "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
DUMMY_KEY = "no_key"

client = OpenAI(base_url=LOCAL_API_BASE, api_key=DUMMY_KEY)

# Local RAG Config
JSON_PATH = "bangladesh_services_data_structure.json"
MD_PATH = "bangladesh_services_data.md"

# Web Search Config
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
CACHE_FILE = "rag_cache.json"
MAX_CHARS = 1000       
MAX_RESULTS = 3       
FAST_TIMEOUT = 5  
ADVANCED_TIMEOUT = 10

# Initialize Cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f: return json.load(f)
    return {}

GLOBAL_CACHE = load_cache()

# ==========================================
# 2. Web Search Logic
# ==========================================
class FastParser:
    @staticmethod
    def clean_html(html):
        if not html: return ""
        tree = HTMLParser(html)
        for tag in tree.css('script, style, nav, footer, header, svg'): tag.decompose()
        text = tree.body.text(separator=' ', strip=True) if tree.body else ""
        return re.sub(r'\s+', ' ', text).strip()[:MAX_CHARS]

async def perform_search(query: str, session: AsyncSession, is_advanced: bool):
    search_depth = "advanced" if is_advanced else "basic"
    timeout = ADVANCED_TIMEOUT if is_advanced else FAST_TIMEOUT
    max_results = 5 if is_advanced else MAX_RESULTS 
    clean_query = f"{query} বাংলাদেশ" if not query.endswith("বাংলাদেশ") else query
    
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": f"{clean_query} latest 2026",
        "search_depth": search_depth, 
        "include_answer": 'basic',
        "max_results": max_results
    }
    
    try:
        resp = await session.post("https://api.tavily.com/search", json=payload, timeout=timeout)
        if resp.status_code == 200:
            res = [{"url": r["url"], "content": FastParser.clean_html(r.get("content", ""))} 
                   for r in resp.json().get("results", [])]
            return sorted(res, key=lambda x: x['url'])
    except Exception as e: 
        print(f"Search error: {e}")
    return []

# ==========================================
# 3. Local PageIndex Logic
# ==========================================
def load_data(json_path, md_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        tree = json.load(f)
    with open(md_path, 'r', encoding='utf-8') as f:
        markdown_lines = f.readlines()
    return tree, markdown_lines

try:
    tree, md_lines = load_data(JSON_PATH, MD_PATH)
    root_structure = tree.get('structure', [])
    print("✅ Loaded document tree structure.")
except Exception as e:
    print(f"⚠️ Error loading data: {e}")
    root_structure, md_lines = [], []

def navigate_tree(query, current_nodes, depth=0):
    if not current_nodes or not isinstance(current_nodes, list): return None
    if len(current_nodes) == 1 and not current_nodes[0].get('nodes'): return current_nodes[0]

    menu = ""
    for idx, node in enumerate(current_nodes):
        title = node.get('title', 'Unknown')
        summary = node.get('summary', 'Category Heading') 
        menu += f"[{idx}] Title: {title}\nSummary/Context: {summary}\n\n"
        
    prompt = f"""You are a precise routing agent navigating a hierarchical Table of Contents for Bangladesh Government Services. 
    Current Options:
    {menu}
    
    User Query: "{query}"
    
    INSTRUCTIONS:
    1. Read the User Query in Bengali.
    2. Mentally translate the query to understand its core topic.
    3. Identify the SINGLE most relevant section index from the Current Options that matches this topic.
    
    Return ONLY the integer index inside the brackets. Do not include any other text.
    """
    
    response = client.chat.completions.create(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    try:
        index_str = "".join(filter(str.isdigit, response.choices[0].message.content))
        selected_node = current_nodes[int(index_str)]
    except (ValueError, IndexError):
        selected_node = current_nodes[0]
        
    if selected_node.get('nodes'):
        return navigate_tree(query, selected_node['nodes'], depth + 1)
    
    return selected_node

def extract_all_leaf_nodes(structure):
    leaves = []
    if isinstance(structure, dict):
        if not structure.get('nodes'): leaves.append(structure)
        else:
            for node in structure.get('nodes', []): leaves.extend(extract_all_leaf_nodes(node))
    elif isinstance(structure, list):
        for item in structure: leaves.extend(extract_all_leaf_nodes(item))
    return leaves

def extract_markdown_text(target_node, root_structure, markdown_lines):
    start_line = target_node.get('line_num', 1) - 1
    all_leaves = extract_all_leaf_nodes(root_structure)
    next_node = next((n for n in all_leaves if n.get('line_num', 0) > target_node.get('line_num', 1)), None)
    end_line = next_node.get('line_num', len(markdown_lines)) - 1 if next_node else len(markdown_lines)
    return "".join(markdown_lines[start_line:end_line])

# ==========================================
# 4. Master QA Bot (RAG -> Web Fallback)
# ==========================================
async def rag_qa_bot(query, history, is_advanced):
    if not root_structure:
        yield "Error: Local data not loaded properly."
        return

    # 1. Start Local RAG
    yield "🔍 **Agent is navigating the local document tree...**"
    target_node = navigate_tree(query, root_structure)
    
    if not target_node:
        yield "⚠️ Could not find a relevant section."
        return
        
    section_title = target_node.get('title', 'Unknown')
    yield f"🔍 **Agent routed to local section:** `{section_title}`\n\n_Reading document..._"

    raw_context = extract_markdown_text(target_node, root_structure, md_lines)

    # 2. Local RAG Prompt with Fallback Trigger
    local_prompt = f"""You are a helpful assistant for Bangladesh Government Services. 
    Answer the user's question based STRICTLY on the provided context. 
    
    CRITICAL INSTRUCTION: If the exact answer is NOT in the context, you MUST output EXACTLY: [SEARCH_REQUIRED]
    Otherwise, answer the question naturally in Bengali (বাংলা).
    
    Context:
    {raw_context}
    
    User Question: {query}
    """
    
    local_response = client.chat.completions.create(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": local_prompt}],
        temperature=0.1,
        stream=True 
    )
    
    final_output = f"🔍 **Local Match:** `{section_title}`\n\n"
    buffer = ""
    search_needed = False

    # 3. Stream Interception Logic
    for chunk in local_response:
        delta = chunk.choices[0].delta.content or ""
        buffer += delta
        
        # We wait until we have a few characters to see if it's triggering the fallback
        
        if "[SEARCH" in buffer:
            search_needed = True
            break
          
        yield final_output+buffer

    # 4. Web Search Fallback Pipeline
    if search_needed:
        mode_text = "Advanced" if is_advanced else "Fast"
        yield f"⚠️ **Not found in local documents.**\n\n🌐 **Triggering {mode_text} Web Search Fallback...**"
        
        # Check Cache First
        hash_input = f"{query.lower().strip()}_{is_advanced}"
        query_hash = hashlib.md5(hash_input.encode()).hexdigest()
        
        if query_hash in GLOBAL_CACHE:
            c = GLOBAL_CACHE[query_hash]
            yield f"🌐 **Web Answer (Cached):**\n\n{c['output']}\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in c['sources']])
            return

        # Perform Search
        async with AsyncSession() as session:
            results = await perform_search(query, session, is_advanced)
        
        if not results:
            yield "❌ **Not found locally, and web search returned no results.**"
            return
            
        context = "\n---\n".join([r["content"] for r in results])
        sources_md = "\n".join([f"- {r['url']}" for r in results])
        yield f"🌐 **Web Search Complete! Synthesizing answer...**\n\n**Sources:**\n{sources_md}\n\n"

        current_time = datetime.now().strftime("%A, %B %d, %Y")
        
        # Web Answer Prompt (From your custom code, adapted for Bengali UI)
        web_prompt = f"""SYSTEM: You are a highly accurate and concise search assistant.
        Current date: {current_time}. 

        CRITICAL INSTRUCTIONS FOR RESPONSE LENGTH:
        CATEGORY 1: Direct Factual Queries (names, dates, numbers) -> Answer in exactly 1 to 2 sentences.
        CATEGORY 2: Complex Explanatory Queries (summaries, reasons) -> Answer in 5 to 10 sentences.
        
        Always reply in Bengali (বাংলা).

        SOURCES:
        {context}

        USER QUERY: {query}
        RESPONSE:"""

        web_response = client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": web_prompt}],
            temperature=0.2,
            stream=True
        )
        
        web_answer = f"🌐 **Web Answer:**\n\n"
        raw_web_text = ""
        for chunk in web_response:
            delta = chunk.choices[0].delta.content or ""
            raw_web_text += delta
            yield web_answer + raw_web_text + f"\n\n**Sources:**\n{sources_md}"

        # Cache the result
        GLOBAL_CACHE[query_hash] = {"output": raw_web_text, "sources": [r['url'] for r in results]}
        with open(CACHE_FILE, 'w') as f: json.dump(GLOBAL_CACHE, f)

# ==========================================
# 5. Launch Gradio App
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🇧🇩 BD Government Services QA (Hybrid RAG)
        Powered by Hierarchical Vectorless Retrieval with an ultra-fast Web Search Fallback agent.
        """
    )
    
    gr.ChatInterface(
        fn=rag_qa_bot,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="সরকারি সেবা বা দেশের খবর সম্পর্কে আপনার প্রশ্ন লিখুন...", container=False, scale=7),
        additional_inputs=[
            gr.Checkbox(label="🔬 Advanced Web Search (if local fails)", value=False)
        ],
        title=None,
        description=None,
        
    )

if __name__ == "__main__":
    print("🚀 Launching Hybrid RAG Gradio Server...")
    demo.launch(server_name="0.0.0.0", server_port=7860)