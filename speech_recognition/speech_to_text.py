# Python program to translate
# speech to text and text to speech


import speech_recognition as sr
import pyttsx3 
import sounddevice

# Initialize the recognizer 
r = sr.Recognizer() 

# Function to convert text to
# speech
def SpeakText(command):
    
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()
    
    
# Loop infinitely for user to
# speak

while(1):    
    
    # Exception handling to handle
    # exceptions at the runtime
    try:
        
        # use the microphone as source for input.
        with sr.Microphone(device_index=25) as source2:
            
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level 
            r.adjust_for_ambient_noise(source2, duration=0.2)
            
            #listens for the user's input 
            audio2 = r.listen(source2)
            
            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            # print(&quot;Did you say &quot;,MyText)
            print(f"Did you say: '{MyText}'?")
            SpeakText(MyText)
            
    except sr.RequestError as e:
        # print(&quot;Could not request results; {0}&quot;.format(e))
        print(f"Could not request results: '{str(e)}'")
        
    except sr.UnknownValueError:
        # print(&quot;unknown error occurred&quot;)
        print(f"unknown error occurred")
