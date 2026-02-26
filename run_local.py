import os
import subprocess



VLLM_BASE_URL = "http://127.0.0.1:5000/v1"
VLLM_API_KEY = "no-key"
VLLM_MODEL_NAME = "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"

def run_pageindex_local(md_path, model_name=VLLM_MODEL_NAME, base_url=VLLM_BASE_URL):
    """
    Executes the PageIndex pipeline routing traffic to a local vLLM server.
    """
    # 1. Inject the environment variables for the local OpenAI server
    env_vars = os.environ.copy()
    env_vars["OPENAI_BASE_URL"] = base_url
    env_vars["CHATGPT_API_KEY"] = VLLM_API_KEY

    # 2. Construct the CLI command
    command = [
        "python3", "run_pageindex.py",
        "--md_path", md_path,
        "--model", model_name,
        "--max-pages-per-node", "5" # Adjust based on your model's context window
    ]

    print(f"🚀 Starting PageIndex...")
    print(f"Targeting Local Model: {model_name}")
    print(f"API Base URL: {base_url}\n")
    
    # 3. Execute the command
    try:
        # check=True ensures an exception is thrown if the script fails
        subprocess.run(command, env=env_vars, check=True, text=True)
        print("\n✅ Extraction complete! Check the output directory for your JSON tree.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ An error occurred during execution: {e}")

if __name__ == "__main__":
    # Define your parameters here
    # PageIndex includes a few test PDFs in the tests/pdfs/ directory
    TARGET_MD = "bangladesh_services_data.md" 
    

    run_pageindex_local(TARGET_MD, VLLM_MODEL_NAME, VLLM_BASE_URL)