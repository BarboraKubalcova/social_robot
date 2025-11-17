import requests
import time
import pyttsx3
import threading
import keyboard
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
    "How does reinforcement learning contribute to adaptive human–robot interaction?",
    "What was the previous question?"
]

session_id = "session_1"  # shared conversation
stop_speech = threading.Event()


def speak(text: str, stop_key: str = "space"):
    """Speak text with interruption on key press."""
    stop_speech.clear()

    def _speak():
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)

        def check_keypress():
            while not stop_speech.is_set():
                if keyboard.is_pressed(stop_key):
                    print(f"Speech interrupted by '{stop_key}' key.")
                    stop_speech.set()
                    engine.stop()
                    break
                time.sleep(0.05)  # small delay to avoid CPU load

        # Start key listener
        key_thread = threading.Thread(target=check_keypress, daemon=True)
        key_thread.start()

        try:
            engine.say(text)
            engine.runAndWait()  # blocks until done or stopped
        except RuntimeError:
            # happens if stopped mid-speech
            pass
        finally:
            engine.stop()

    t = threading.Thread(target=_speak)
    t.start()
    t.join()


def main_voice_loop(language="sk-SK"):

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
            mode = resp.json()["mode"]
            print(f"Answer is: {answer}")
            # print(f"Sources are: {sources}")
            print(f"Mode: {mode}\n")

            speak(answer)

        except sr.UnknownValueError:
            print("Could not understand speech.")
        except sr.RequestError as e:
            print(f"⚠Speech recognition error: {e}")
        except Exception as e:
            print(f"⚠Unexpected error: {e}")

        # Small delay between cycles
        time.sleep(0.3)


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

    main_voice_loop(language="en-US")


if __name__ == "__main__":
    main()
