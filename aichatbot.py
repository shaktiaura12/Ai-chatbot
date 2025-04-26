import openai
import speech_recognition as sr
import pyttsx3
import tensorflow as tf
import librosa
import numpy as np

openai.api_key = "your_openai_api_key"
emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful']

emotion_model = tf.keras.models.load_model("emotion_model.h5")

engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

def extract_features(audio_data_list, sr):
    audio_np = np.array(audio_data_list, dtype=np.float32)
    mfccs = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed.reshape(1, -1)

def detect_emotion(audio_data_list, sr):
    features = extract_features(audio_data_list, sr)
    prediction = emotion_model.predict(features)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

def record_audio_as_list(duration=4, sample_rate=22050):
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=sample_rate) as source:
        print("🎤 Recording for emotion detection...")
        audio = recognizer.record(source, duration=duration)
        audio_data = audio.get_raw_data()
        audio_list = list(np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0)
    return audio_list, sample_rate

def get_text_from_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🗣️ Say something...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand you."

def get_chat_response(prompt, emotion):
    full_prompt = f"The user is feeling {emotion}. Respond appropriately.\nUser: {prompt}\nAI:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=full_prompt,
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def chat_loop():
    while True:
        audio_list, sr = record_audio_as_list()
        emotion = detect_emotion(audio_list, sr)
        print(f"🧠 Detected Emotion: {emotion}")

        user_text = get_text_from_speech()
        print(f"👤 You said: {user_text}")

        response = get_chat_response(user_text, emotion)
        print(f"🤖 AI: {response}")
        speak(response)

if __name__ == "__main__":
    chat_loop()
