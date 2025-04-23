# Who A(m) I?
# Text-Image Feedback Loop

A system that creates a feedback loop between text and images using Ollama and Replicate.

## Features
- Text expansion using Ollama's gnokit/improve-prompt model
- Image generation with Stable Diffusion via Replicate
- Image description using LLaVA via Replicate
- Interactive Streamlit interface for easy use

## Requirements
- Python 3.8+
- Ollama with gnokit/improve-prompt model
- Replicate API token
- Required Python packages: streamlit, requests, openai, pillow

## Setup and Installation
1. Clone this repository
2. Install required packages: `pip install streamlit requests openai pillow`
3. Install Ollama and pull the gnokit/improve-prompt model
4. Get a Replicate API token from [replicate.com](https://replicate.com)
5. Run the Streamlit app: `streamlit run llm_ollama_stable_diffusion_streamlit.py`

## How It Works
1. Enter a short text prompt
2. The system expands it into a detailed description
3. An image is generated from this description
4. The image is analyzed to create a new text description
5. This process repeats for the specified number of iterations
