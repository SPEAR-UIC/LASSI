import subprocess
import html
import re
import json
from IPython.display import display, HTML


def argo_fetch_with_curl(data):
    url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/streamchat/"
    
    # Add a header stating that the content type is JSON
    headers = [
        "-H", "Content-Type: application/json",
        "-H", "Accept: text/plain",  # Expect standard text streaming
        "-H", "Accept-Encoding: identity",  # Prevent gzip compression
        "-H", "Host: apps-dev.inside.anl.gov",
    ]

    # Create an HTML display object for live streaming output
    html_display = display(HTML(""), display_id=True)

    # Run curl as a subprocess, capturing output as it streams
    process = subprocess.Popen(
        ["curl", "-k", "-X", "POST", url, *headers, "-d", data],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line-buffered for real-time output
    )

    response_text = ""

    # Read output line by line to maintain real-time streaming
    for line in iter(process.stdout.readline, ''):
        # Append new content and escape HTML characters for safety
        response_text += line
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
        
        # Create typewriter effect with a blinking cursor
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

    process.stdout.close()
    process.wait()  # Ensure process exits cleanly

    return response_text  # Store full response for later use


def argo_llm(model, system_prompt, content_prompt, temp=0.2, verifyprompt=False):  
    
    if model == "gpto1preview":
        content_prompt = system_prompt + "\n\n" + content_prompt
    
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
        "user": "mdearing",
        "model": model,
        "system": system_prompt,
        "prompt": [content_prompt],
        "stop": [],
        "temperature": round(temp, 1),
        "top_p": 0.9,
    }

    # Convert the dict to JSON string for curl
    payload = json.dumps(data)

    response = argo_fetch_with_curl(payload)

    return response