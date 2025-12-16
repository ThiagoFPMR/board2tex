# board2tex

board2tex is a Python tool designed to convert images of whiteboard drawings and equations into LaTeX and TikZ code. It uses OpenCV for image processing to segment individual components from a board image and leverages Google's Gemini AI to transcribe these segments into high-quality LaTeX code.

## Features

- **Board Parsing**: Automatically detects and segments individual drawings or equations from a whiteboard image using computer vision techniques.
- **AI Transcription**: Uses the Gemini API to accurately convert visual content into LaTeX/TikZ code.
- **Automated Workflow**: Streamlines the process from raw image to ready-to-use LaTeX files.

## Missing Features
- [ ] Handling board reflections
- [ ] Ability to run as a server
- [ ] Graphical user interface (_streamlit_)
- [ ] Ability to select board in broader picture and adjust POV
- [ ] Ability to select content of interest in the board
- [ ] Upgrade separation by color and make it optional
