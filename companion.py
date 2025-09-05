# AI Seeing Companion - Voice-Controlled Smart Glasses Assistant
# Fully accessible for blind users with voice commands and audio interface

import cv2
import numpy as np
import easyocr
import pyttsx3
import threading
import time
from datetime import datetime
import requests
from io import BytesIO
from PIL import Image
from collections import defaultdict, deque
import queue
import speech_recognition as sr
import os
import sys

class VoiceControlledAICompanion:
    def __init__(self):
        """
        Voice-controlled AI Companion for blind users
        Everything is controlled through voice commands and audio feedback
        """
        print("🎤 Initializing Voice-Controlled AI Companion...")
        
        # Camera source (will be set through voice)
        self.use_esp32 = False
        self.esp32_url = ""
        self.cap = None
        
        # Voice recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise (important for accuracy)
        print("🎵 Calibrating microphone for ambient noise... Please stay quiet for 2 seconds.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Microphone calibrated!")
        
        # Initialize OCR
        self.ocr_reader = None  # Will initialize after voice setup
        
        # Initialize Text-to-Speech with natural voice
        self.tts_engine = pyttsx3.init()
        self.setup_natural_voice()
        
        # YOLO model
        self.yolo_net = None
        self.yolo_classes = []
        self.yolo_output_layers = []
        
        # Control variables
        self.is_running = False
        self.speech_queue = queue.Queue()
        self.listening_for_commands = True
        
        # Smart announcement system
        self.stable_objects = defaultdict(int)
        self.last_announcement = time.time()
        self.announcement_interval = 10  # Adjustable through voice
        self.last_text_announcement = time.time()
        
        # Movement detection
        self.previous_frame = None
        self.movement_threshold = 25000
        
        # Voice command recognition
        self.command_keywords = {
            'describe': ['describe', 'what do you see', 'tell me', 'look around'],
            'read_text': ['read text', 'read', 'any text', 'what does it say'],
            'find_person': ['find person', 'anyone there', 'people around'],
            'find_object': ['find', 'where is', 'look for'],
            'settings': ['settings', 'options', 'configure', 'setup'],
            'help': ['help', 'commands', 'what can you do'],
            'pause': ['pause', 'stop talking', 'quiet'],
            'resume': ['resume', 'continue', 'start'],
            'exit': ['exit', 'quit', 'goodbye', 'stop']
        }
        
        print("✅ AI Companion ready for voice setup!")
    
    def setup_natural_voice(self):
        """Configure TTS for natural, pleasant voice"""
        self.tts_engine.setProperty('rate', 160)
        self.tts_engine.setProperty('volume', 0.9)
        
        # Try to find the best available voice
        voices = self.tts_engine.getProperty('voices')
        if voices:
            for voice in voices:
                # Prefer female voices or specific good voices
                if any(keyword in voice.name.lower() for keyword in ['zira', 'female', 'hazel', 'susan']):
                    self.tts_engine.setProperty('voice', voice.id)
                    break
    
    def speak_and_wait(self, text):
        """Speak text and wait for completion (blocking)"""
        print(f"🔊 {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def speak_async(self, text):
        """Add text to speech queue (non-blocking)"""
        if text:
            self.speech_queue.put(text)
    
    def listen_for_voice_command(self, timeout=5, phrase_time_limit=3):
        """Listen for voice commands with timeout"""
        try:
            with self.microphone as source:
                print("🎤 Listening...")
                # Shorter timeout for better responsiveness
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            try:
                command = self.recognizer.recognize_google(audio).lower()
                print(f"🎙️ Heard: {command}")
                return command
            except sr.UnknownValueError:
                return ""
            except sr.RequestError:
                self.speak_and_wait("Voice recognition service unavailable. Please try again.")
                return ""
                
        except sr.WaitTimeoutError:
            return ""
    
    def voice_setup_wizard(self):
        """Voice-guided setup wizard for blind users"""
        self.speak_and_wait("Welcome to AI Seeing Companion! I am your digital eyes. Let's set up your camera.")
        
        # Camera source selection
        self.speak_and_wait("Say 'webcam' to use your phone or computer camera, or say 'glasses' to use ESP32 smart glasses.")
        
        while True:
            command = self.listen_for_voice_command(timeout=10)
            
            if 'webcam' in command or 'phone' in command or 'camera' in command:
                self.use_esp32 = False
                self.speak_and_wait("Using webcam or phone camera. Initializing...")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.speak_and_wait("Camera not found. Please ensure your camera is connected and try again.")
                    continue
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                break
                
            elif 'glasses' in command or 'esp' in command:
                self.use_esp32 = True
                self.speak_and_wait("Using ESP32 smart glasses. Please tell me the IP address of your glasses.")
                self.speak_and_wait("For example, say: 'one nine two dot one six eight dot one dot one zero zero'")
                
                ip_command = self.listen_for_voice_command(timeout=15)
                if ip_command:
                    # Convert voice IP to proper format
                    ip_address = self.parse_voice_ip(ip_command)
                    if ip_address:
                        self.esp32_url = f"http://{ip_address}:81/shot.jpg"
                        self.speak_and_wait(f"Connecting to glasses at {ip_address}")
                        
                        # Test connection
                        if self.test_esp32_connection():
                            self.speak_and_wait("Successfully connected to your smart glasses!")
                            break
                        else:
                            self.speak_and_wait("Cannot connect to glasses. Please check the IP address and try again.")
                    else:
                        self.speak_and_wait("Could not understand IP address. Please try again.")
                else:
                    self.speak_and_wait("No IP address received. Trying webcam instead.")
                    self.use_esp32 = False
                    self.cap = cv2.VideoCapture(0)
                    break
            else:
                self.speak_and_wait("Please say 'webcam' for phone camera or 'glasses' for ESP32 smart glasses.")
        
        # Initialize AI models
        self.speak_and_wait("Setting up AI vision. This may take a moment...")
        self.initialize_ai_models()
        
        # Quick tutorial
        self.speak_and_wait("Setup complete! Here are voice commands you can use:")
        time.sleep(1)
        self.speak_and_wait("Say 'describe' to hear what I see.")
        time.sleep(0.5)
        self.speak_and_wait("Say 'read text' to read any text.")
        time.sleep(0.5)
        self.speak_and_wait("Say 'find person' to locate people nearby.")
        time.sleep(0.5)
        self.speak_and_wait("Say 'help' anytime for more commands.")
        time.sleep(0.5)
        self.speak_and_wait("Say 'exit' to stop the companion.")
        time.sleep(1)
        self.speak_and_wait("I'm ready to be your digital eyes! I'll continuously monitor your surroundings and respond to your voice commands.")
    
    def parse_voice_ip(self, voice_input):
        """Convert voice IP address to proper format"""
        # Replace common voice patterns
        voice_input = voice_input.replace('dot', '.')
        voice_input = voice_input.replace('point', '.')
        voice_input = voice_input.replace(' ', '')
        
        # Try to extract IP pattern
        import re
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        match = re.search(ip_pattern, voice_input)
        
        if match:
            return match.group()
        
        # Manual number conversion for common patterns
        number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }
        
        for word, digit in number_words.items():
            voice_input = voice_input.replace(word, digit)
        
        # Try again
        match = re.search(ip_pattern, voice_input)
        return match.group() if match else None
    
    def test_esp32_connection(self):
        """Test ESP32-CAM connection"""
        try:
            response = requests.get(self.esp32_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def initialize_ai_models(self):
        """Initialize OCR and object detection models"""
        try:
            # Initialize OCR
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            
            # Load YOLO model
            self.download_yolo_files()
            self.yolo_net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
            
            with open("coco.names", "r") as f:
                self.yolo_classes = [line.strip() for line in f.readlines()]
            
            layer_names = self.yolo_net.getLayerNames()
            self.yolo_output_layers = [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
            
        except Exception as e:
            self.speak_and_wait(f"Warning: Some AI features may be limited due to setup error: {str(e)}")
    
    def download_yolo_files(self):
        """Download YOLO model files with progress"""
        import os
        import urllib.request
        
        files = {
            "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
            "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
            "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
        }
        
        for filename, url in files.items():
            if not os.path.exists(filename):
                self.speak_and_wait(f"Downloading AI model file: {filename}")
                try:
                    urllib.request.urlretrieve(url, filename)
                except Exception as e:
                    self.speak_and_wait(f"Download failed: {e}")
    
    def get_frame(self):
        """Get frame from camera source"""
        if self.use_esp32:
            try:
                response = requests.get(self.esp32_url, timeout=3)
                image = Image.open(BytesIO(response.content))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                return True, frame
            except:
                return False, None
        else:
            if self.cap:
                return self.cap.read()
            return False, None
    
    def perform_ocr(self, frame):
        """Extract text from frame"""
        if not self.ocr_reader:
            return []
        
        try:
            results = self.ocr_reader.readtext(frame)
            texts = [text for (_, text, confidence) in results if confidence > 0.6]
            return texts
        except:
            return []
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        if not self.yolo_net:
            return []
        
        try:
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.yolo_net.setInput(blob)
            outputs = self.yolo_net.forward(self.yolo_output_layers)
            
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            detected_objects = []
            if len(boxes) > 0:
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                if len(indexes) > 0:
                    for i in indexes.flatten():
                        label = str(self.yolo_classes[class_ids[i]])
                        detected_objects.append(label)
            
            return detected_objects
            
        except:
            return []
    
    def process_voice_command(self, command):
        """Process voice commands"""
        command = command.lower()
        
        # Check each command category
        for cmd_type, keywords in self.command_keywords.items():
            if any(keyword in command for keyword in keywords):
                if cmd_type == 'describe':
                    self.describe_scene()
                elif cmd_type == 'read_text':
                    self.read_text()
                elif cmd_type == 'find_person':
                    self.find_people()
                elif cmd_type == 'find_object':
                    self.find_specific_object(command)
                elif cmd_type == 'settings':
                    self.voice_settings()
                elif cmd_type == 'help':
                    self.voice_help()
                elif cmd_type == 'pause':
                    self.pause_companion()
                elif cmd_type == 'resume':
                    self.resume_companion()
                elif cmd_type == 'exit':
                    return False
                return True
        
        self.speak_and_wait("I didn't understand that command. Say 'help' for available commands.")
        return True
    
    def describe_scene(self):
        """Describe current scene"""
        ret, frame = self.get_frame()
        if not ret:
            self.speak_and_wait("Cannot access camera.")
            return
        
        objects = self.detect_objects(frame)
        if objects:
            unique_objects = list(set(objects))
            description = f"I can see: {', '.join(unique_objects[:6])}"
            self.speak_and_wait(description)
        else:
            self.speak_and_wait("I don't see any clear objects in the current view.")
    
    def read_text(self):
        """Read text in current view"""
        ret, frame = self.get_frame()
        if not ret:
            self.speak_and_wait("Cannot access camera.")
            return
        
        texts = self.perform_ocr(frame)
        if texts:
            combined_text = '. '.join(texts[:3])  # Read first 3 texts
            self.speak_and_wait(f"Text detected: {combined_text}")
        else:
            self.speak_and_wait("No readable text found in current view.")
    
    def find_people(self):
        """Look for people in the scene"""
        ret, frame = self.get_frame()
        if not ret:
            self.speak_and_wait("Cannot access camera.")
            return
        
        objects = self.detect_objects(frame)
        people_count = objects.count('person')
        
        if people_count > 0:
            if people_count == 1:
                self.speak_and_wait("I can see one person nearby.")
            else:
                self.speak_and_wait(f"I can see {people_count} people nearby.")
        else:
            self.speak_and_wait("I don't see any people in the current view.")
    
    def find_specific_object(self, command):
        """Find a specific object mentioned in the command"""
        # Extract object name from command
        object_to_find = None
        for obj_class in self.yolo_classes:
            if obj_class in command:
                object_to_find = obj_class
                break
        
        if not object_to_find:
            self.speak_and_wait("Please specify what object you want me to find.")
            return
        
        ret, frame = self.get_frame()
        if not ret:
            self.speak_and_wait("Cannot access camera.")
            return
        
        objects = self.detect_objects(frame)
        if object_to_find in objects:
            count = objects.count(object_to_find)
            if count == 1:
                self.speak_and_wait(f"Yes, I can see a {object_to_find}.")
            else:
                self.speak_and_wait(f"Yes, I can see {count} {object_to_find}s.")
        else:
            self.speak_and_wait(f"I don't see any {object_to_find} in the current view.")
    
    def voice_settings(self):
        """Voice-controlled settings"""
        self.speak_and_wait("Settings menu. Say 'speech speed' to change how fast I talk, or 'announcement frequency' to change how often I describe scenes automatically.")
        
        command = self.listen_for_voice_command(timeout=10)
        
        if 'speech' in command and 'speed' in command:
            self.speak_and_wait("Say 'faster' or 'slower'")
            speed_cmd = self.listen_for_voice_command()
            if 'faster' in speed_cmd:
                current_rate = self.tts_engine.getProperty('rate')
                self.tts_engine.setProperty('rate', min(current_rate + 20, 200))
                self.speak_and_wait("Speech speed increased.")
            elif 'slower' in speed_cmd:
                current_rate = self.tts_engine.getProperty('rate')
                self.tts_engine.setProperty('rate', max(current_rate - 20, 100))
                self.speak_and_wait("Speech speed decreased.")
        
        elif 'announcement' in command or 'frequency' in command:
            self.speak_and_wait("Say 'more frequent' for more announcements or 'less frequent' for fewer announcements.")
            freq_cmd = self.listen_for_voice_command()
            if 'more' in freq_cmd:
                self.announcement_interval = max(self.announcement_interval - 3, 5)
                self.speak_and_wait("I'll make announcements more frequently.")
            elif 'less' in freq_cmd:
                self.announcement_interval += 3
                self.speak_and_wait("I'll make announcements less frequently.")
    
    def voice_help(self):
        """Provide voice help"""
        help_text = """
        Here are the voice commands you can use:
        
        Say 'describe' or 'what do you see' - I'll tell you what's in view.
        Say 'read text' - I'll read any text I can see.
        Say 'find person' or 'anyone there' - I'll look for people.
        Say 'find' followed by an object name - I'll look for that object.
        Say 'settings' - Change speech speed or announcement frequency.
        Say 'pause' - I'll stop automatic announcements.
        Say 'resume' - I'll restart automatic announcements.
        Say 'exit' or 'goodbye' - I'll shut down.
        
        I'm always listening for your commands while monitoring your surroundings.
        """
        self.speak_and_wait(help_text)
    
    def pause_companion(self):
        """Pause automatic announcements"""
        self.listening_for_commands = False
        self.speak_and_wait("Automatic announcements paused. I'm still listening for your voice commands.")
    
    def resume_companion(self):
        """Resume automatic announcements"""
        self.listening_for_commands = True
        self.speak_and_wait("Automatic announcements resumed.")
    
    def speech_worker(self):
        """Background speech worker"""
        while self.is_running:
            try:
                text = self.speech_queue.get(timeout=1)
                if text:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    self.speech_queue.task_done()
            except queue.Empty:
                continue
    
    def continuous_monitoring(self):
        """Continuous scene monitoring with voice command listening"""
        frame_count = 0
        last_objects = []
        
        while self.is_running:
            ret, frame = self.get_frame()
            if not ret:
                time.sleep(1)
                continue
            
            frame_count += 1
            
            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                current_objects = self.detect_objects(frame)
                
                # Detect significant changes
                if self.listening_for_commands:
                    new_objects = [obj for obj in current_objects if obj not in last_objects]
                    important_new = [obj for obj in new_objects if obj in ['person', 'car', 'truck', 'bicycle']]
                    
                    if important_new:
                        alert = f"Alert: {', '.join(important_new)} detected"
                        self.speak_async(alert)
                    
                    # Regular scene updates
                    if time.time() - self.last_announcement > self.announcement_interval and current_objects:
                        unique_objects = list(set(current_objects))[:4]
                        if unique_objects:
                            scene_desc = f"Environment: {', '.join(unique_objects)}"
                            self.speak_async(scene_desc)
                            self.last_announcement = time.time()
                
                last_objects = current_objects
            
            # Listen for voice commands (non-blocking)
            try:
                with self.microphone as source:
                    # Very short listen to catch commands without blocking
                    audio = self.recognizer.listen(source, timeout=0.1, phrase_time_limit=1)
                    try:
                        command = self.recognizer.recognize_google(audio).lower()
                        if command:
                            if not self.process_voice_command(command):
                                break
                    except (sr.UnknownValueError, sr.RequestError):
                        pass
            except sr.WaitTimeoutError:
                pass
            
            time.sleep(0.1)  # Small delay
    
    def run(self):
        """Main run method"""
        try:
            # Voice setup
            self.voice_setup_wizard()
            
            # Start speech worker
            self.is_running = True
            speech_thread = threading.Thread(target=self.speech_worker, daemon=True)
            speech_thread.start()
            
            # Start continuous monitoring
            self.speak_and_wait("Starting continuous monitoring. I'm ready to assist you!")
            self.continuous_monitoring()
            
        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.speak_and_wait(f"An error occurred: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.speak_and_wait("AI Companion shutting down. Goodbye!")

def main():
    """Main function with error handling for different platforms"""
    print("🤖 AI SEEING COMPANION - Voice Controlled")
    print("=" * 50)
    
    # Check if running on phone (Termux) or desktop
    if 'ANDROID_ROOT' in os.environ:
        print("📱 Detected Android/Termux environment")
        print("Make sure you have granted microphone and camera permissions!")
    
    try:
        companion = VoiceControlledAICompanion()
        companion.run()
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        print("Please check your microphone and camera permissions.")

if __name__ == "__main__":
    main()