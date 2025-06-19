# app.py

import os
import whisper
from gtts import gTTS
#import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
   
from transformers import BitsAndBytesConfig

# --- CONFIG ---
AUDIO_DURATION = 5  # seconds to record
LANG = "it"
MODEL_NAME = "sapienzanlp/Minerva-7B-instruct-v1.0"
WHISPER_MODEL = "small"  # Whisper model size

import os

def get_audio_files(directory="wav"):
    """
    Returns a list of full paths to .wav files in the specified directory.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return []

    audio_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".wav")
    ]
    
    if not audio_files:
        print("No .wav files found.")
    return audio_files


whisper_model = whisper.load_model(WHISPER_MODEL)
# --- STEP 2: Transcribe user audio ---
def transcribe_audio(audio_path):
    print("Trascrizione in corso...")
    result = whisper_model.transcribe(audio_path, language=LANG)
    print(result['text'])
    return result['text']

def load_minerva_pipeline():
    print("Caricamento LLM locale in 8-bit...")

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto"  # Will use GPU if available, else CPU
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    return HuggingFacePipeline(pipeline=pipe)


# # --- STEP 4: Build LangChain conversation --- ---> con history
# def build_conversation(llm):
#     prompt = PromptTemplate(
#         input_variables=["history", "input"], 
#         template="""Tu sei un assistente vocale intelligente. Rispondi alle domande che ti vengono poste.

# {history}
# Utente: {input}
# Assistente:"""
#     )
#     memory = ConversationBufferMemory()
#     return ConversationChain(prompt=prompt, llm=llm, memory=memory)

def build_prompt_wrapper(llm):
    prompt = PromptTemplate(
        input_variables=["input"],
        template="""Sei un assistente virtuale intelligente. Rispondi alla seguente domanda in modo conciso:

Domanda: {input}
Risposta:"""
    )

    def run(query):
        prompt_text = prompt.format(input=query)
        return llm(prompt_text)
    
    return run



# --- STEP 5: Speak response ---
def speak_text(text):
    print("Risposta:", text)
 #   tts = gTTS(text=text, lang=LANG)
 #   tts.save("response.mp3")
  #  os.system("start response.mp3")  # Windows; use 'afplay' for macOS

# --- MAIN LOOP ---
def main():
    llm = load_minerva_pipeline()
    convo = build_prompt_wrapper(llm)
    
    audio_paths = get_audio_files()
    for path in audio_paths:
        query = transcribe_audio(path)
        
        if not query.strip():
            print("⚠️ Nessuna voce rilevata.")
            continue

        #response = convo.run(query) -> con history
        response = convo(query)

        print("==========================================")
        speak_text(response)
        print("==========================================")

if __name__ == "__main__":
    main()
