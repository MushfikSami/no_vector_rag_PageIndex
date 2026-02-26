import os
import json
import gradio as gr
from openai import OpenAI

# ==========================================
# 1. Configuration & Setup
# ==========================================
LOCAL_API_BASE = "http://localhost:5000/v1"
LOCAL_MODEL = "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
DUMMY_KEY = "local_vllm_dummy_key"

client = OpenAI(base_url=LOCAL_API_BASE, api_key=DUMMY_KEY)

JSON_PATH = "bangladesh_services_data_structure.json"
MD_PATH = "bangladesh_services_data.md"

# ==========================================
# 2. Core PageIndex Logic
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
    """Recursively navigate the tree by asking the LLM to choose the best sub-branch."""
    # Base case: We reached a leaf node
    if not current_nodes or not isinstance(current_nodes, list):
        return None
        
    # If the current node is a leaf (no 'nodes' key), return it
    if len(current_nodes) == 1 and not current_nodes[0].get('nodes'):
        return current_nodes[0]

    # Create a menu of the current level's options
    menu = ""
    for idx, node in enumerate(current_nodes):
        title = node.get('title', 'Unknown')
        # If it's a category, it might not have a summary, so just use the title
        summary = node.get('summary', 'Category Heading') 
        menu += f"[{idx}] Title: {title}\nSummary/Context: {summary}\n\n"
        
    prompt = f"""You are a precise routing agent. 
    Below are the options at the current level of a document's Table of Contents.
    Read the user's query and identify the SINGLE most relevant section index (the number in the brackets) to explore further.
    
    Current Options:
    {menu}
    
    User Query: "{query}"
    
    Return ONLY the integer index inside the brackets. Do not include any other text or explanation.
    """
    
    response = client.chat.completions.create(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    try:
        # Extract the index
        index_str = "".join(filter(str.isdigit, response.choices[0].message.content))
        selected_idx = int(index_str)
        selected_node = current_nodes[selected_idx]
    except (ValueError, IndexError):
        # Fallback to the first option if the LLM fails to format the number
        selected_node = current_nodes[0]
        
    # If the selected node has children, recurse deeper
    if selected_node.get('nodes'):
        print(f"  ↳ Navigating deeper into: {selected_node.get('title')}")
        return navigate_tree(query, selected_node['nodes'], depth + 1)
    
    # Otherwise, we found our leaf node
    return selected_node

def find_next_leaf_node(structure, current_line_num):
    """Helper to find the line number of the next section to know where to stop reading."""
    all_leaves = extract_all_leaf_nodes(structure)
    for node in all_leaves:
        if node.get('line_num', 0) > current_line_num:
            return node
    return None

def extract_all_leaf_nodes(structure):
    """Used only for boundary detection, not for prompting."""
    leaves = []
    if isinstance(structure, dict):
        if not structure.get('nodes'): 
            leaves.append(structure)
        else:
            for node in structure.get('nodes', []):
                leaves.extend(extract_all_leaf_nodes(node))
    elif isinstance(structure, list):
        for item in structure:
            leaves.extend(extract_all_leaf_nodes(item))
    return leaves

def extract_markdown_text(target_node, root_structure, markdown_lines):
    start_line = target_node.get('line_num', 1) - 1
    next_node = find_next_leaf_node(root_structure, target_node.get('line_num', 1))
    end_line = next_node.get('line_num', len(markdown_lines)) - 1 if next_node else len(markdown_lines)
    return "".join(markdown_lines[start_line:end_line])

# ==========================================
# 3. Gradio Interface Logic
# ==========================================
def rag_qa_bot(query, history):
    if not root_structure:
        yield "Error: Data not loaded properly. Check your file paths."
        return

    yield "🔍 **Agent is navigating the document tree...**"

    # Step A: Traverse the Tree
    target_node = navigate_tree(query, root_structure)
    
    if not target_node:
        yield "⚠️ Could not find a relevant section."
        return
        
    section_title = target_node.get('title', 'Unknown')
    yield f"🔍 **Agent routed to section:** `{section_title}`\n\n_Reading document..._"

    # Step B: Extract Context
    raw_context = extract_markdown_text(target_node, root_structure, md_lines)

    # Step C: Stream the Answer
    prompt = f"""You are a helpful assistant for Bangladesh Government Services. 
    Answer the user's question based STRICTLY on the provided context. 
    If the answer is not in the context, say "I could not find the answer in the provided documents."
    Always reply in Bengali (বাংলা).
    
    Context:
    {raw_context}
    
    User Question: {query}
    """
    
    response = client.chat.completions.create(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=True 
    )
    
    final_answer = f"🔍 **Agent routed to section:** `{section_title}`\n\n"
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            final_answer += chunk.choices[0].delta.content
            yield final_answer

# ==========================================
# 4. Launch Gradio App
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🇧🇩 BD Government Services QA (PageIndex RAG)
        Ask questions about birth registration, health services, and more. 
        Powered by Hierarchical Vectorless Reasoning Retrieval.
        """
    )
    
    gr.ChatInterface(
        fn=rag_qa_bot,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="সরকারি সেবা সম্পর্কে আপনার প্রশ্ন লিখুন... (উদাঃ ডেঙ্গু পরীক্ষা কোথায় বিনামূল্যে করা যায়?)", container=False, scale=7),
        title=None,
        description=None
    )

if __name__ == "__main__":
    print("🚀 Launching Gradio Server...")
    demo.launch(share=True)