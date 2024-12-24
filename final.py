import streamlit as st
import wave
import pyaudio
import requests
from PIL import Image
import easyocr
import numpy as np
import google.generativeai as genai
import os
import re

# Configure Gemini API
genai.configure(api_key="AIzaSyAjhjE1-c6vcFixyO6lOIHQUE8a15peRd0")
model = genai.GenerativeModel("gemini-1.5-flash")

def record_audio(filename, duration=5, rate=44100, chunk=1024):
    """Records audio from the microphone and saves it as a .wav file."""
    audio = pyaudio.PyAudio()

    # Open the stream
    stream = audio.open(format=pyaudio.paInt16, 
                        channels=1, 
                        rate=rate, 
                        input=True, 
                        frames_per_buffer=chunk)
    
    st.write(f"Recording for {duration} seconds...")
    frames = []

    # Record the audio
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    st.write("Recording complete.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a .wav file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

def send_to_sarvam_api(filepath):
    """Sends the recorded audio file to the Sarvam AI API for speech-to-text translation."""
    url = "https://api.sarvam.ai/speech-to-text-translate"
    
    headers = {
        'api-subscription-key': '44de06bc-2820-4709-9f01-b60acff28d0f'
    }
    
    payload = {
        'model': 'saaras:v1',
        'prompt': ''  # Empty prompt if not required
    }

    # Open the audio file in binary mode
    with open(filepath, 'rb') as audio_file:
        files = [
            ('file', (filepath, audio_file, 'audio/wav'))
        ]
        response = requests.post(url, headers=headers, data=payload, files=files)
    
    if response.status_code == 200:
        # Parse and return only the transcript
        response_data = response.json()  # Convert response text to JSON
        if "transcript" in response_data:
            return response_data["transcript"]
        else:
            return "Transcript key not found in response."
    else:
        return f"Failed to transcribe speech. HTTP Status Code: {response.status_code}"

def generate_response(input_text, chat_history=None):
    try:
        # Combine chat history and current prompt
        if chat_history:
            input_text = "\n".join(chat_history) + f"\nUser: {input_text}\nBot:"

        # Modify the input text to include more bank-specific instructions
        input_text = f"""
        You are a helpful customer care assistant for a bank. You assist customers with various banking issues, including errors in transaction messages, instructions on how to fill challenges (such as forms, documents), and general inquiries about banking operations. 

        If the customer provides an error message, explain what the error means, what caused it, and guide the user on how to resolve it. If the error cannot be solved directly, suggest who they should approach or what actions they should take.

        If the user asks about forms, instructions, or any challenges related to banking (like how to fill in a form or document), provide clear and concise instructions, step-by-step guidance, and common pitfalls to avoid.

        Always maintain a friendly, professional, and clear tone. If any sensitive information is provided, ensure that privacy is respected (e.g., mask bank account numbers).

        The user may also ask about common banking procedures, and you should provide detailed explanations for those as well.

        Current user query: {input_text}
        """

        # Call the model to generate the response
        response = model.generate_content(input_text)
        return response.text if response.text else "No response generated."
    
    except Exception as e:
        st.error(f"Error: {e}")
        return "Error: Could not generate a response."


def protect_sensitive_info(text):
    """
    Function to protect sensitive information like bank account numbers
    """
    def mask_bank_account(match):
        # Extract the bank account number
        bank_acc_text = match.group(1)
        bank_acc_num = match.group(2)
        
        # Keep first 4 characters of the account number, mask the rest
        visible_part = bank_acc_num[:4]
        masked_part = '*' * (len(bank_acc_num) - 4)
        
        # Reconstruct the string with masked account number
        return f"{bank_acc_text}{visible_part + masked_part}"
    
    # Regex pattern to match 'bankacc' followed by digits
    bank_acc_pattern = r'(bankacc)(\d+)'
    
    # Apply privacy protection
    protected_text = re.sub(bank_acc_pattern, mask_bank_account, text)
    
    return protected_text

def extract_text_from_image(image):
    try:
        # Convert the PIL image to a NumPy array
        image_array = np.array(image)
        
        # Initialize easyOCR reader for English
        reader = easyocr.Reader(["en"])  
        result = reader.readtext(image_array, detail=0)  # Extract text without bounding box info
        
        # Join the extracted text
        extracted_text = "\n".join(result).strip()
        
        # Apply privacy protection
        protected_text = protect_sensitive_info(extracted_text)
        
        return protected_text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return "Error: Could not extract text."

def main():
    st.title("Customer Care and Doubt Clarification")
    st.write("Chat using text, speech, or image input!")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Sidebar for image and speech input
    st.sidebar.header("Input Options")
    
    # Image Upload
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    # Speech Input
    st.sidebar.subheader("Speech Input")
    if st.sidebar.button("Start Recording"):
        # Create a temporary file for audio recording
        audio_file = "recorded_audio.wav"
        
        # Record audio
        record_duration = st.sidebar.slider("Recording Duration", 1, 10, 5)
        record_audio(audio_file, duration=record_duration)
        
        # Transcribe speech
        transcribed_text = send_to_sarvam_api(audio_file)
        
        # Remove the temporary audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        # If transcription successful, set the transcribed text as the prompt
        if transcribed_text:
            st.session_state['speech_input'] = transcribed_text
            st.sidebar.success(f"Transcribed Text: {transcribed_text}")

    # Text Input Area
    prompt = st.text_input("Enter your prompt:", 
                           value=st.session_state.get('speech_input', ''))

    # Apply privacy protection to input prompt
    protected_prompt = protect_sensitive_info(prompt)

    # Process Image if uploaded
    extracted_text = ""
    if uploaded_image is not None:
        st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        image = Image.open(uploaded_image).convert("RGB")
        extracted_text = extract_text_from_image(image)
        st.write("**Extracted Text from Image:**")
        st.write(extracted_text)

    # Generate Response Button
    if st.button("Send"):
        # Prepare input text
        if extracted_text:
            # Combine extracted text and prompt
            combined_input = f"Image context: {extracted_text}\nUser prompt: {protected_prompt}"
        else:
            combined_input = protected_prompt

        # Get bot response
        response = generate_response(combined_input, st.session_state["chat_history"])
        
        # Update chat history
        st.session_state["chat_history"].append(f"User: {protected_prompt}")
        st.session_state["chat_history"].append(f"Bot: {response}")
        
        # Display response
        st.write("**Bot Response:**")
        st.write(response)

    # Display Chat History
    if st.session_state["chat_history"]:
        st.write("**Chat History:**")
        for chat in st.session_state["chat_history"]:
            st.write(chat)

# Run the app
if __name__ == "__main__":
    main()