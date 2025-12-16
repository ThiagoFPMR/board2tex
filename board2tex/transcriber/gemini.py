"""
Implementation of the Transcriber interface using Google's Gemini API.
"""

import numpy as np
from PIL import Image
from google import genai

from transcriber.interface import Transcriber


class GeminiTranscriber(Transcriber):
    def __init__(self, prompt: str, api_key: str, model: str = "gemini-3-pro-preview", **kwargs):
        super().__init__(prompt)
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def transcribe_to_tex(self, image: np.ndarray) -> str:
        """
        Transcribes the given image to LaTeX code using the Gemini API.

        Args:
            image (np.ndarray): The input image as a NumPy array.
        Returns:
            str: The transcribed LaTeX code.
        """
        # Create a request to the GenAI API
        pil_image = Image.fromarray(image)
        # 3. Call the API
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[pil_image, self.prompt]
            )
            return response.text
        except Exception as e:
            return f"Error calling Gemini API: {e}"