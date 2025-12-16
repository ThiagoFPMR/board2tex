"""
Interface for transcribing images to LaTeX code using VLMs.
"""

import numpy as np
from abc import ABC, abstractmethod


class Transcriber(ABC):
    def __init__(self, prompt:str, **kwargs):
        """
        Transcriber interface constructor.

        Args:
            prompt (str): The prompt to guide the transcription process.
        """
        self.prompt = prompt

    @abstractmethod
    def transcribe_to_tex(self, image: np.ndarray) -> str:
        """
        Transcribes the given image to LaTeX code.

        Args:
            image (np.ndarray): The input image as a NumPy array.
        Returns:
            str: The transcribed LaTeX code.
        """
        pass