import os
import streamlit as st
import requests
import json
import base64
import io
import time
import uuid
import replicate
from PIL import Image
from openai import OpenAI

class FeedbackLoop:
    def __init__(self, ollama_url="http://localhost:11434", openai_api_key=None, replicate_api_token=None, output_dir=None):
        # Set up Replicate API
        self.replicate_api_token = replicate_api_token or os.environ.get("REPLICATE_API_TOKEN")
        if self.replicate_api_token:
            os.environ["REPLICATE_API_TOKEN"] = self.replicate_api_token
        
        # Set up OpenAI client (keeping for backwards compatibility)
        self.openai_api_key = openai_api_key
        if openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
        
        # Set up Ollama API
        self.ollama_url = ollama_url
        
        # Set up output directories
        self.output_dir = output_dir or os.path.join(os.path.expanduser("~"), "feedback-loop-project", "output")
        self.images_dir = os.path.join(self.output_dir, "images")
        self.prompts_dir = os.path.join(self.output_dir, "prompts")
        
        # Create output directories if they don't exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.prompts_dir, exist_ok=True)

    def expand_prompt(self, short_prompt, status_placeholder):
        status_placeholder.write("Expanding the prompt...")
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "gnokit/improve-prompt",
                    "prompt": short_prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                expanded_prompt = response.json().get("response", "").strip()
                
                # Save the prompt
                prompt_path = os.path.join(self.prompts_dir, f"prompt_{uuid.uuid4()}.txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(expanded_prompt)
                
                status_placeholder.write("✅ Prompt expanded successfully")
                return expanded_prompt
            else:
                status_placeholder.write("⚠️ Error with Ollama. Falling back to Replicate...")
                return self._expand_prompt_with_replicate(short_prompt, status_placeholder)
        except Exception as e:
            status_placeholder.write(f"⚠️ Exception: {str(e)}. Falling back to Replicate...")
            return self._expand_prompt_with_replicate(short_prompt, status_placeholder)
    
    def _expand_prompt_with_replicate(self, short_prompt, status_placeholder):
        try:
            # Using Llama 3 on Replicate for fallback prompt expansion
            output = replicate.run(
                "meta/llama-3-8b-instruct:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",
                input={
                    "prompt": f"You are a creative prompt engineer for image generation. Expand this short prompt into a detailed and vivid scene description including style, lighting, mood, and composition. Just provide the expanded prompt without explanations: {short_prompt}"
                }
            )
            
            # Replicate returns output as a generator, collect all parts
            result = ""
            for item in output:
                result += item
                
            expanded_prompt = result.strip()
            
            # Save the prompt
            prompt_path = os.path.join(self.prompts_dir, f"prompt_{uuid.uuid4()}.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(expanded_prompt)
            
            status_placeholder.write("✅ Prompt expanded with Replicate")
            return expanded_prompt
        except Exception as e:
            status_placeholder.write(f"❌ Error expanding with Replicate: {str(e)}")
            return short_prompt
    
    def generate_image(self, prompt, status_placeholder):
        status_placeholder.write("Generating image from prompt using Replicate...")
        
        try:
            # Using Stable Diffusion on Replicate - Updated model version ID
            output = replicate.run(
                "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
                input={
                    "prompt": prompt,
                    "width": 768,
                    "height": 768,
                    "num_outputs": 1,
                    "scheduler": "K_EULER_ANCESTRAL",
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "seed": 42,
                    "negative_prompt": "ugly, blurry, poor quality, deformed, disfigured",
                }
            )
            
            # Output contains image URLs
            if output and len(output) > 0:
                image_url = output[0]
                
                # Download the image
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image = Image.open(io.BytesIO(image_response.content))
                    
                    # Save the image
                    image_path = os.path.join(self.images_dir, f"image_{uuid.uuid4()}.png")
                    image.save(image_path)
                    
                    status_placeholder.write("✅ Image generated successfully with Replicate")
                    return image, image_path
                else:
                    status_placeholder.write(f"❌ Failed to download image from URL")
                    return None, None
            else:
                status_placeholder.write("❌ No image URL returned from Replicate")
                return None, None
        except Exception as e:
            status_placeholder.write(f"❌ Error generating image: {str(e)}")
            return None, None
    
    def describe_image(self, image, status_placeholder):
        status_placeholder.write("Generating description from image using Replicate...")
        
        try:
            # Save image temporarily to upload to Replicate
            temp_image_path = os.path.join(self.images_dir, f"temp_{uuid.uuid4()}.png")
            image.save(temp_image_path)
            
            # Using LLaVA on Replicate for image description
            output = replicate.run(
                "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
                input={
                    "image": open(temp_image_path, "rb"),
                    "prompt": "Describe this image in detail as if you were creating a prompt for an image generator. Be creative and focus on visual elements, style, mood, and atmosphere. Do not start with phrases like 'This image shows' or 'I can see'. Just describe the content directly."
                }
            )
            
            # Clean up temporary file
            os.remove(temp_image_path)
            
            # Replicate returns output as a generator, collect all parts
            result = ""
            for item in output:
                result += item
                
            description = result.strip()
            
            # Save the description
            description_path = os.path.join(self.prompts_dir, f"description_{uuid.uuid4()}.txt")
            with open(description_path, "w", encoding="utf-8") as f:
                f.write(description)
            
            status_placeholder.write("✅ Image description generated successfully")
            return description
        except Exception as e:
            status_placeholder.write(f"❌ Error describing image: {str(e)}")
            return None

def main():
    st.title("🔄 Text-Image Feedback Loop")
    
    # Add custom CSS for better appearance
    st.markdown("""
    <style>
        .stTextInput > div > div > input {
            font-size: 20px;
        }
        .stButton > button {
            width: 100%;
        }
        .step-header {
            background-color: #4e8cff;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
        }
        .output-area {
            border: 1px solid #f0f2f6;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        .iteration-separator {
            height: 2px;
            background-color: #e0e0e0;
            margin: 30px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initial prompt input
    initial_prompt = st.text_input("Enter your initial prompt:", 
                                   placeholder="A cat playing piano")
    
    # Iterations slider
    iterations = st.slider("Number of Iterations", 1, 5, 3)
    
    if st.button("Start Feedback Loop"):
        if not initial_prompt:
            st.error("Please enter an initial prompt")
            return
        
        # Ask for API token if not set in environment
        if not os.environ.get("REPLICATE_API_TOKEN"):
            replicate_api_token = st.text_input("Enter your Replicate API Token:", type="password")
            if not replicate_api_token:
                st.error("Please enter your Replicate API Token")
                return
        else:
            replicate_api_token = None  # Will use the environment variable
        
        # Initialize feedback loop
        loop = FeedbackLoop(replicate_api_token=replicate_api_token)
        
        # Results container
        results_container = st.container()
        
        with results_container:
            current_prompt = initial_prompt
            
            for i in range(iterations):
                st.markdown(f"<h2>Iteration {i+1}</h2>", unsafe_allow_html=True)
                
                # Prompt Expansion
                st.markdown("<div class='step-header'>Step 1: Expanding Prompt</div>", unsafe_allow_html=True)
                status_placeholder = st.empty()
                expanded_prompt = loop.expand_prompt(current_prompt, status_placeholder)
                
                with st.expander("View Expanded Prompt", expanded=True):
                    st.markdown(f"<div class='output-area'>{expanded_prompt}</div>", unsafe_allow_html=True)
                
                # Image Generation
                st.markdown("<div class='step-header'>Step 2: Generating Image</div>", unsafe_allow_html=True)
                status_placeholder = st.empty()
                image, image_path = loop.generate_image(expanded_prompt, status_placeholder)
                if image:
                    st.image(image, caption=f"Generated Image - Iteration {i+1}", use_column_width=True)
                
                # Image Description
                st.markdown("<div class='step-header'>Step 3: Describing Image</div>", unsafe_allow_html=True)
                status_placeholder = st.empty()
                if image:
                    new_prompt = loop.describe_image(image, status_placeholder)
                    with st.expander("View Image Description", expanded=True):
                        st.markdown(f"<div class='output-area'>{new_prompt}</div>", unsafe_allow_html=True)
                    
                    # Update current prompt for next iteration
                    current_prompt = new_prompt
                else:
                    st.error("Could not generate an image. Stopping loop.")
                    break
                
                # Add separator between iterations
                if i < iterations - 1:
                    st.markdown("<div class='iteration-separator'></div>", unsafe_allow_html=True)
            
            st.success("Feedback loop complete!")
            st.balloons()

if __name__ == "__main__":
    main()