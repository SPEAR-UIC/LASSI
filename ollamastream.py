import ollama
import subprocess
import html
import re
import json
from IPython.display import display, HTML


def unload_ollama_model(model):
    print("\nUnloading Ollama server...")
    data_curl = {
        "model": model,
        "keep_alive": 0
    }
    data_curl_json = json.dumps(data_curl)
    curl_command_unload_LLM = [
        'curl',
        'http://localhost:11434/api/generate',
        '-d',
        data_curl_json
    ]
    result_unload_LLM = subprocess.run(curl_command_unload_LLM, capture_output=True, text=True)
    print("\nUnload LLM:\n", result_unload_LLM.stdout)
    print(result_unload_LLM.stderr)


def ollama_streaming(data):
       
    # Create a display object to show streaming text
    output_display = display(display_id=True)
    
    # Create an HTML display object with a <pre> tag for formatted text
    html_display = display(HTML(""), display_id=True)
    
    response_text = ""
    stream_response = ollama.chat(
        model=data["model"],
        messages=[
            {"role": "system", "content": data["system"]},
            {'role': 'user', 'content': data["prompt"]}
        ],
        stream=True,
        options=dict(
            temperature=data["temperature"],
            top_p=data["top_p"]
        )
    )
    
    # Process each chunk in the stream
    for chunk in stream_response:
        # print(chunk['message']['content'], end='', flush=True)
        chunk_content = chunk['message']['content']
        response_text += chunk_content
        escaped_text = html.escape(response_text)

        # (Optional) If you want to add specific styling to tags, e.g. <think> tags:
        styled_text = re.sub(
            r'&lt;think&gt;',
            '<span style="color: #d32f2f; font-weight: bold;">&lt;think&gt;</span>',
            escaped_text
        )
        styled_text = re.sub(
            r'&lt;/think&gt;',
            '<span style="color: #d32f2f; font-weight: bold;">&lt;/think&gt;</span>',
            styled_text
        )

        # Create typewriter effect with cursor
        display_text = f"{styled_text}<span style='border-right:2px solid;animation:blink 1s infinite'> </span>"

        # Update the display output with styled text
        # Use dedent to remove any leading whitespace from the HTML template
        # Put everything on one line to avoid extra whitespace
        html_content = (
            f"<pre style='background:#f8f8f8;padding:10px;border-radius:5px;border:1px solid #ddd;"
            f"white-space:pre-wrap;font-family:monospace;margin:0;'>{display_text}</pre>"
            f"<style>"
            f"@keyframes blink {{0%{{opacity:1}}50%{{opacity:0}}100%{{opacity:1}}}}"
            f"</style>"
        )
        
        # Update the display output with styled text
        html_display.update(HTML(html_content))
    
    # Clear the streaming display after completion
    # clear_output(wait=True)

    return response_text


"""
Local hosted LLM via OLLAMA API, GGUF quantized models

Requirement: ollama serve [from server command line]
Requirement: ollama.pull('modelname') [from Python script to pre-load selected model at the server]

ollama.list() [preview what is available to use on the server]

Example: model='codestral:latest'
"""
def ollama_llm(model, system_prompt, content_prompt, temp=0.2, verifyprompt=False):
    # Ollama serve must already be running on a port on this server 
    
    if verifyprompt:
        print("\n=====> VERIFY THE PROMPTING <=====\n")
        print(f"TEMPERATURE: {round(temp, 1)}")
        print(f"\nSYSTEM:\n {system_prompt}")
        print(f"\nPROMPT:\n {content_prompt}")
        print("\n=====^^^^^ END PROMPT REVIEW ^^^^^=====\n")
    else:
        print("\n=====< LLM Prompt Not Shown >=====\n")
    
    # Data to be sent as a POST in JSON format
    data = {
        "model": model,
        "system": system_prompt,
        "prompt": content_prompt,
        "stop": [],
        "temperature": round(temp, 1),
        "top_p": 0.9,
    }

    response = ollama_streaming(data)
    
    return response