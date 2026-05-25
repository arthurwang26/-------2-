import os
import time
from utils.logger import get_logger
import google.generativeai as genai

logger = get_logger("gemini_video")

class GeminiVideoCaptioner:
    def __init__(self, api_key: str, model_name="gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.model = None

    def load(self):
        if self.model is not None:
            return
        logger.info(f"Initializing Gemini API ({self.model_name})...")
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name=self.model_name)
            logger.info("Gemini API initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            self.model = None

    def generate_video_caption(self, video_path: str) -> str:
        """Uploads video to Gemini and gets a temporal action caption."""
        if self.model is None:
            return "Gemini API not loaded."

        try:
            logger.info(f"Uploading {video_path} to Gemini...")
            video_file = genai.upload_file(path=video_path)
            
            # Wait for processing
            while video_file.state.name == "PROCESSING":
                logger.info("Waiting for Gemini video processing...")
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
                
            if video_file.state.name == "FAILED":
                logger.error("Gemini video processing failed.")
                return "Gemini processing failed."
                
            prompt = (
                "Please watch this video and describe the pure physical sequence of actions of the people in it. "
                "CRITICAL RULES: Do NOT describe facial expressions. Do NOT guess emotions or feelings (e.g., do not say 'calm' or 'anxious'). "
                "Only describe objective physical facts like 'sitting down', 'holding a cup', 'walking', 'talking'. "
                "Provide a brief, concise summary of the continuous action."
            )
            
            logger.info("Generating content with Gemini...")
            response = self.model.generate_content([video_file, prompt])
            
            # Clean up file from Google's servers
            genai.delete_file(video_file.name)
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Gemini API Error: {e}"
