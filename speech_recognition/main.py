import speech_recognition as sr
import pyttsx3 
import sounddevice
import pyaudio
import requests

# Function to convert text to
# speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()


def get_llm_response(prompt):
    url = "http://localhost:5000/generate"
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result.get("generated_text", "No response from LLM")
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    
    
def get_speech(r, mic_id):
        
    try:
        with sr.Microphone(device_index=mic_id) as source2:
            print("Listening...")
            r.adjust_for_ambient_noise(source2, duration=1)
            
            #listens for the user's input 
            audio2 = r.listen(source2, timeout=None)
            
            # Using google to recognize audio
            spoken_text = r.recognize_google(audio2, language="sk-SK")
            spoken_text = spoken_text.lower()
            print(f"Did you say: '{spoken_text}'?")
        return spoken_text
            
    except sr.RequestError as e:
        print(f"Could not request results: '{str(e)}'")
    except sr.UnknownValueError:
        print(f"unknown error occurred")



def get_mic_id():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    mic_id = None

    for i in range(0, numdevices):
            device_name = p.get_device_info_by_host_api_device_index(0, i).get('name')
            max_input_channel = p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')
            # if max_input_channel > 0:
            #     print(f"Input Device id {i} - {device_name}")
            if "ReSpeaker" in device_name and max_input_channel > 0:
                    mic_id = i
    if not mic_id:
        raise Exception("No Respeaker microphone detected.")
     
    return mic_id



def main():
    r = sr.Recognizer() 
    
    mic_id = get_mic_id() # also checks if mic is connected
    print(f"ReSpeaker microphone id is: {mic_id}")
    text = get_speech(r, mic_id) 
    response = get_llm_response(text)
    print("LLM Response:", response)


if __name__ == "__main__":
    main()


"""
https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/#extract-voice
"""
