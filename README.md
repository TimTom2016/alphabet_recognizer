# Alphabet Recognizer

A browser-based alphabet recognizer that allows users to draw letters directly in the browser, which are then classified by a trained model. The project also includes training utilities to improve or retrain the model using custom data.

---

## ✨ Features

- ✍️ Draw letters directly in the browser
- 🧠 Recognize hand-drawn alphabet characters
- 🛠️ Train your own model using Rust and WGPU

---

## 🧰 Prerequisites

- **Rust** (with `cargo`) — [Install Rust](https://www.rust-lang.org/tools/install)
- **just** — For command automation
  ```bash
  cargo install just
  ```
- **Python 3** — Used for running the local development server

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/timtom2016/alphabet_recognizer.git
cd alphabet_recognizer
```

### 2. Train the Model (Optional but Recommended)

#### 2.1. Prepare Training Images

Create a folder named `Letter/` in the project root with the following structure:

```
Letter/
├── A/
├── B/
├── C/
...
```

Each subfolder should contain PNG images representing that letter. These images will be used for training.

#### 2.2. Run the Training Script

```bash
just train
```

This runs the training package using the `wgpu` feature.

#### 2.3. Move the Trained Model

After training, a `model.bin` file will be created at:

```
/tmp/alphabet_recognizer/model.bin
```

Move it to the web directory so it can be used by the browser app:

```bash
mv /tmp/alphabet_recognizer/model.bin web/model.bin
```

---

### 3. Build the Web App

```bash
just build_web
```

This command:
- Navigates to the `web/` directory
- Executes `build-for-web.sh` to compile the app for the browser using WGPU

---

### 4. Run the Web App Locally

```bash
just run_web
```

This command:
- Starts a local development server
- Open the app at `http://localhost:8000`
- Your Browser needs to support wgpu: [Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)
---

## 📁 Project Structure

```
alphabet_recognizer/
├── shared/              # Shared model definitions
├── train/               # Rust training logic
├── web/                 # Web interface (WASM + HTML/JS)
├── justfile             # Task shortcuts
├── README.md            # This file
```

---

## 🧪 Notes

- Make sure your GPU drivers are up to date to avoid WGPU runtime issues.
- Training time may vary depending on your dataset and hardware.

---

## 📜 License

MIT License

---
