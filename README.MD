#  CleanTok: TikTok Watermark Remover 🎥❌

A mobile application that allows users to upload TikTok videos and remove watermarks using either **blurring** or **inpainting**—your choice! Designed for flexibility and ease of use, this app uses **React Native with Expo** for the frontend and a **Flask** backend with OpenCV and FFmpeg for the watermark removal process.

---

## 📦 Features

- ✅ Upload or select an MP4 video
- 🔍 Detects watermark(s) using OpenCV template matching
- 🎯 Removes watermark using your choice of **blur** or **inpaint**
- 📱 Mobile-friendly UI built with React Native
- 💡 Option to process videos with multiple or moving watermarks

---

## 🚀 Demo

Here’s a quick look at the app in action:

![Demo Screenshot](./tiktok_logo.png)

---

## 🛠️ Installation and Setup

### 1. Backend (Flask API)

#### Prerequisites:
- Python 3.8+
- FFmpeg installed and added to PATH
- `virtualenv` recommended

#### Installation:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

#### Running the Backend:
```bash
python app.py
```

#### ⚠️ IMPORTANT:
You **must update the backend URL** in the `frontend/app.json` file to match your backend server:
```json
"extra": {
  "API_URL": "http://<your-ip>:5000"
}
```
Replace `<your-ip>` with your actual IP address or `localhost` if running locally.

---

### 2. Frontend (React Native with Expo)

#### Prerequisites:
- Node.js & npm/yarn
- Expo CLI: Install it globally using `npm install -g expo-cli`

#### Installation:
```bash
cd frontend
npm install  # or yarn install
```

#### Running the App:
```bash
npx expo start
```

You can then run the app on an emulator or scan the QR code using Expo Go on your phone.

---

## 🧰 Libraries Used

### Backend:
- **OpenCV**: For watermark detection and template matching.
- **Flask**: Lightweight REST API framework.
- **FFmpeg**: used to apply `delogo` filter for blurring or masking.
- **NumPy**: Image processing and matrix operations

### Frontend:
- **React Native**: Cross-platform mobile app development
- **Expo AV / Document Picker / Font / Constants**: Core utilities for media and platform handling
- **React Navigation & Paper**: For routing and UI design

---

## ⚖️ Trade-Offs and Assumptions

- Assumes watermark is present in standard TikTok positions (top-left or bottom-right).
- Template matching may fail if the watermark is semi-transparent or animated.
- FFmpeg blur is fast but not as visually clean as inpainting; user can choose.
- Focused on performance—uses efficient processing steps to minimize lag on mobile devices.

---

## 🧪 Testing

You can verify your setup with test video called `test_video.mp4`. Run the script:

```bash
python test_watermark_removal.py
```

Make sure FFmpeg is installed and in your system path.

---

## ✨ BONUS FEATURES

I’ve implemented the following smart upgrades:

- 🎯 **Supports multiple watermark positions** (e.g., TikTok logo jumps between corners)
- 🧠 **Inpainting or Blur configurable via dropdown**
- 📹 **Handles 360p to 1080p** resolution videos
- ⚡ **Optimized for low-resource devices** (especially during FFmpeg filtering)
- 🔄 **Scalable architecture** to allow batching in the future
- ✅ **Basic testing script** included for watermark detection/removal

---

## 💡 Future Improvements

Some ideas to make this solution even better:

- ☁️ Cloud based video processing using AWS Lambda or Google Cloud Functions
- 🔁 Caching previously processed videos to avoid reprocessing
- 🧠 Use of deep learning for more accurate watermark detection

---
