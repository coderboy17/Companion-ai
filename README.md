👓 AI Seeing Companion - Voice-Controlled Smart Glasses Assistant

A voice-controlled AI companion designed for blind and visually impaired users, acting as digital eyes using computer vision, OCR, and speech recognition.
The assistant works fully through voice commands and provides audio feedback.

It supports:

Object detection (using YOLOv4-Tiny)

Text recognition (OCR with EasyOCR)

Voice-guided camera setup (webcam or ESP32-CAM smart glasses)

Real-time scene description

Person detection

Smart voice-based settings and help

🚀 Features

🎤 Voice-First Interface – Fully operable via voice commands

👁️ Scene Description – Describes objects and surroundings

📖 Text Reader – Reads aloud any visible text

🧍 Person Finder – Detects people in the environment

🔍 Object Finder – Locates specific objects on request

⚙️ Voice Settings – Change speech speed and announcement frequency

🔄 Continuous Monitoring – Alerts for new important objects (e.g., people, cars)

📡 ESP32-CAM Integration – Supports smart glasses camera input

📦 Requirements

Make sure you have the following installed:

Python 3.7+

Required libraries (install via pip):

pip install opencv-python numpy easyocr pyttsx3 SpeechRecognition pillow requests


YOLOv4-tiny model files (downloaded automatically at runtime if missing):

yolov4-tiny.cfg

yolov4-tiny.weights

coco.names

🖥️ Usage
1. Clone the Repository
git clone https://github.com/your-username/ai-seeing-companion.git
cd ai-seeing-companion

2. Run the Assistant
python ai_companion.py

3. Voice Setup Wizard

When launched, the assistant will guide you via speech:

Say "webcam" → to use computer/phone camera

Say "glasses" → to use ESP32-CAM smart glasses (you’ll provide IP address by voice)

🎙️ Voice Commands

Here are the main commands you can say anytime:

Command	Action
"describe" / "what do you see"	Describes the current scene
"read text"	Reads text from the camera feed
"find person"	Detects nearby people
"find [object]"	Looks for a specific object (e.g., "find chair")
"settings"	Opens voice settings menu
"pause"	Pauses automatic announcements
"resume"	Resumes automatic announcements
"help"	Lists available commands
"exit" / "goodbye"	Shuts down the assistant
📱 ESP32-CAM Smart Glasses

If using ESP32 smart glasses:

Connect ESP32-CAM to WiFi

Note the IP address (e.g., 192.168.1.100)

During setup, say: "glasses" → then speak the IP address in natural form (e.g., "one nine two dot one six eight dot one dot one zero zero")

⚠️ Notes

Ensure microphone and camera permissions are granted

Works on Windows, Linux, macOS, and Android (Termux)

Uses Google Speech Recognition API (requires internet connection)

YOLOv4-tiny is optimized for real-time detection on low-resource devices

🛠️ Future Improvements

Add offline speech recognition (Vosk or Whisper)

Support for multi-language OCR

Expand object detection models for better accuracy

Mobile-friendly packaging (APK / standalone app)

👤 Author

Developed as an AI accessibility project for visually impaired assistance
