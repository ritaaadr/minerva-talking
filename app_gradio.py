import gradio as gr
import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from transformers import BitsAndBytesConfig

LANG = "it"
WHISPER_MODEL = "small"
LLM_MODEL = "sapienzanlp/Minerva-7B-instruct-v1.0"

# Load Whisper and Minerva
whisper_model = whisper.load_model(WHISPER_MODEL)

quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    quantization_config=quant_config,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
    )

llm = HuggingFacePipeline(pipeline=pipe)

def transcribe_and_respond(audio):
    text = whisper_model.transcribe(audio)["text"]
    prompt = f"Tu sei un assistente vocale intelligente.Rispondi in modo preciso e non continuare la conversazione\nUtente: {text}\nAssistente:"
    
    raw_output = llm(prompt)
    # raw_output is a list of strings; take the first one and clean it
    response = raw_output[0] if isinstance(raw_output, list) else str(raw_output)
    response = response.replace(prompt, "").strip()
    return f"🗣️ Domanda: {text}\n Risposta: {response}"

demo = gr.Interface(fn=transcribe_and_respond, 
                    inputs=gr.Audio(type="filepath", label="🎙️ Registra la tua voce"),
                    outputs="text",
                    title="Voice Assistant (Whisper + Minerva)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
