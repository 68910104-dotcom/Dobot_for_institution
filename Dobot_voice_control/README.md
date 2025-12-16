# 🤖 Dobot Voice Control System (Thai/English Supported)

This project controls a **Dobot Magician** robotic arm using voice commands. It features a hybrid language system (Thai & English), fuzzy logic for error correction, and a teach-and-replay memory mode.

## ✨ Features

* **🎙️ Voice Control:** Control X, Y, Z, R axes movement via microphone.
* **🇹🇭 Hybrid Language:** Supports both **Thai** and **English** commands simultaneously.
* **🧠 Smart Fuzzy Logic:** Automatically corrects misheard words (e.g., *Leaf* $\rightarrow$ **Left**, *ทราย* $\rightarrow$ **ซ้าย**).
* **💨 Suction System:** Toggle the suction cup (Air Pump) ON/OFF.
* **💾 Memory Mode (Teach & Play):**
    * **Save:** Record the current coordinate.
    * **Play:** Replay the sequence of recorded coordinates (Automation).
    * **Clear:** Reset the memory.

## 🛠️ Hardware Requirements

1.  **Dobot Magician** (Robotic Arm)
2.  **Suction Cup Kit** (Air pump + Suction head)
3.  **Computer/Laptop** (Windows recommended)
4.  **Microphone**
5.  **USB Cable**

## ⚙️ Installation

1.  **Prerequisites:**
    * Python 3.10 or 3.11 is recommended.
    * Git (Optional)

2.  **Install Dependencies:**
    ```bash
    pip install pydobot pyserial SpeechRecognition sounddevice numpy scipy
    ```

3.  **Driver Setup:**
    * Ensure the **Silicon Labs CP210x USB to UART Bridge** driver is installed.

4.  **Configuration:**
    * Open `voice_control.py`.
    * Edit the `PORT` variable to match your Dobot's port (Check Device Manager).
    ```python
    PORT = "COM3"  # Example: COM3, COM4
    ```

## 🎮 Usage

Run the main script:
```bash
python voice_control.py
