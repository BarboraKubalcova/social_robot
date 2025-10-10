import requests
import time
import pyttsx3
import speech_recognition as sr

# questions = [
#     "What is Retrieval-Augmented Generation (RAG) and how does it enhance AI-Generated Content?",
#     "What are the key limitations of RAG systems?",
#     "How is RAG applied across different AI modalities?",
#     "What are some potential future directions for RAG research?",
#     "What are the different types of retrievers used in RAG?",
#     "What are the four stages of competence in skill development?",
#     "What is the main difference between social robots and service robots?",
#     "How does reinforcement learning contribute to adaptive human–robot interaction?",
#     "What are the advantages of using cloud computing in robotics research?",
#     "What are the main goals of the PhD thesis discussed in the document?"
# ]

questions = [
    "What is Retrieval-Augmented Generation (RAG) and how does it enhance AI-Generated Content?",
    "What was the previous question?"
]

session_id = "session_1"  # shared conversation

# Initialize TTS engine
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("volume", 1.0)  # volume [0.0, 1.0]
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 200)


def speak(text: str):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()
    # for voice in voices:
    #     engine.setProperty('voice', voice.id)
    #     print(voice.id)
    #     engine.say('The quick brown fox jumped over the lazy dog.')
    # engine.runAndWait()


def speech_to_text(mic, recognizer, timeout: int = 5, phrase_time_limit: int = 60,
                   language: str = "en-US") -> str | None:
    with mic as source:
        print("🎤 Listening... (say something)")
        recognizer.adjust_for_ambient_noise(source, duration=0.6)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("⏰ Listening timed out while waiting for speech.")
            return None

    try:
        text = recognizer.recognize_google(audio, language=language)
        return text.lower()
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
    except sr.RequestError as e:
        print(f"Error with the speech recognition service; {e}")

    return None



def main():
    # for question in questions:
    #     print(f"Question: {question}")
    #     start_time = time.time()
    #     resp = requests.post(
    #         "http://127.0.0.1:8000/search_docs",
    #         json={"query": question, "session_id": session_id}
    #     )
    #     print(f"mcp speed: {(time.time() - start_time)}s")
    #     answer = resp.json()["answer"]
    #     print(answer)
    #     speak(answer)

    language = "sk-SK"
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # Calibrate ambient noise once at startup
    with mic as source:
        print("🛠️ Calibrating ambient noise... please stay quiet.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        # recognizer.energy_threshold = energy_threshold
        recognizer.dynamic_energy_threshold = True

        print(f"Energy threshold set to: {recognizer.energy_threshold}")
        print("Ready — I’m listening continuously (say 'stop' to quit).\n")

    while True:
        with mic as source:
            # Wait until recognizer detects speech energy above threshold
            print("Waiting for speech...")
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)

        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio, language=language)
            print(f"You said: {text}\n\n")

            text = text.lower()
            if "stop" in text:
                print("Stopping listening.")
                break

            start_time = time.time()
            resp = requests.post(
                "http://127.0.0.1:8000/search_docs",
                json={"query": text, "session_id": session_id}
            )
            print(f"mcp speed: {(time.time() - start_time)}s")
            answer = resp.json()["answer"]
            sources = resp.json()["sources"]
            print(f"Answer is: {answer}")
            print(f"Sources are: {sources}")

            speak(answer)

        except sr.UnknownValueError:
            print("Could not understand speech.")
        except sr.RequestError as e:
            print(f"⚠Speech recognition error: {e}")
        except Exception as e:
            print(f"⚠Unexpected error: {e}")

        # Small delay between cycles
        time.sleep(0.3)


if __name__ == "__main__":
    main()
