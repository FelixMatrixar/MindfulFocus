import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any
import json

class LLaVAMedicalAnalyzer:
   def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"):
       self.device = "cuda" if torch.cuda.is_available() else "cpu"
       print(f"Loading LLaVA model on {self.device}...")
       
       self.processor = LlavaNextProcessor.from_pretrained(model_name)
       self.model = LlavaNextForConditionalGeneration.from_pretrained(
           model_name,
           torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
           low_cpu_mem_usage=True,
           device_map="auto" if self.device == "cuda" else None
       )
       
       if self.device == "cpu":
           self.model = self.model.to(self.device)
       
       print("LLaVA model loaded successfully!")
   
   def analyze_facial_asymmetry(self, image: np.ndarray, metrics_json: str) -> str:
       """Analyze facial asymmetry using hybrid approach (image + structured data)"""
       
       # Convert numpy array to PIL Image
       if isinstance(image, np.ndarray):
           if len(image.shape) == 3 and image.shape[2] == 3:
               # Convert BGR to RGB
               image_rgb = image[:, :, ::-1]
           else:
               image_rgb = image
           pil_image = Image.fromarray(image_rgb.astype(np.uint8))
       else:
           pil_image = image
       
       prompt = f"""<|user|>
<image>
You are a medical AI assistant specializing in facial landmark analysis. Analyze the attached facial image for signs of neurological conditions, particularly focusing on facial asymmetry that might indicate conditions like Bell's palsy or other cranial nerve disorders.

To assist your analysis, here is precise landmark data extracted from this image:

{metrics_json}

Please provide a detailed assessment that:
1. Describes any visible asymmetry in the face
2. Correlates the visual observations with the provided metrics
3. Identifies which facial regions (upper, middle, lower) show the most significant asymmetry
4. Suggests possible clinical considerations based on the pattern of asymmetry
5. Rates the overall severity on a scale of 0-10 (0 = perfectly symmetric, 10 = severe asymmetry)

Focus on medical accuracy and be specific about anatomical landmarks and measurements.<|assistant|>"""

       inputs = self.processor(pil_image, prompt, return_tensors="pt").to(self.device)
       
       with torch.no_grad():
           output = self.model.generate(
               **inputs,
               max_new_tokens=500,
               do_sample=True,
               temperature=0.7,
               pad_token_id=self.processor.tokenizer.eos_token_id
           )
       
       response = self.processor.decode(output[0], skip_special_tokens=True)
       
       # Extract just the assistant's response
       if "<|assistant|>" in response:
           response = response.split("<|assistant|>")[-1].strip()
       
       return response
   
   def analyze_iris_features(self, image: np.ndarray, iris_coords: Dict[str, Any]) -> str:
       """Analyze iris features for medical indicators"""
       
       if isinstance(image, np.ndarray):
           if len(image.shape) == 3 and image.shape[2] == 3:
               image_rgb = image[:, :, ::-1]
           else:
               image_rgb = image
           pil_image = Image.fromarray(image_rgb.astype(np.uint8))
       else:
           pil_image = image
       
       prompt = f"""<|user|>
<image>
You are a medical AI assistant specializing in iris analysis. Examine the iris regions in this facial image for potential health indicators.

The iris centers are located at these coordinates:
- Left iris center: {iris_coords['left']}
- Right iris center: {iris_coords['right']}

Please analyze:
1. Iris color and uniformity
2. Any visible patterns, spots, or discoloration
3. Pupil size and symmetry
4. Any signs of corneal arcus, Kayser-Fleischer rings, or other notable features
5. Overall iris health assessment

Provide specific medical observations based on what you can see in the iris regions.<|assistant|>"""

       inputs = self.processor(pil_image, prompt, return_tensors="pt").to(self.device)
       
       with torch.no_grad():
           output = self.model.generate(
               **inputs,
               max_new_tokens=400,
               do_sample=True,
               temperature=0.7,
               pad_token_id=self.processor.tokenizer.eos_token_id
           )
       
       response = self.processor.decode(output[0], skip_special_tokens=True)
       
       if "<|assistant|>" in response:
           response = response.split("<|assistant|>")[-1].strip()
       
       return response
