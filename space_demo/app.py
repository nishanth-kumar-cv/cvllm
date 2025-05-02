import gradio as gr
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

base_model = "mistralai/Mistral-7B-v0.1"
peft_repo = "cognitivevideos/cvLLM"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, peft_repo)

def generate_script(prompt, tone):
    input_text = f"Generate a {tone} product video script: {prompt}"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

with gr.Blocks() as demo:
    gr.Markdown("## cvLLM: Cognitive Videos LLM Demo")
    with gr.Row():
        prompt = gr.Textbox(label="What should the script be about?", placeholder="e.g. A smart water bottle")
        tone = gr.Dropdown(["professional", "friendly", "humorous"], value="professional", label="Tone")
    output = gr.Textbox(label="Generated Script")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=generate_script, inputs=[prompt, tone], outputs=output)

demo.launch()