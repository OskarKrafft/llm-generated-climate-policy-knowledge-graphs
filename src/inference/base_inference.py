import os
import json
import requests
from abc import ABC, abstractmethod

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate(self, messages, max_tokens=2000, temperature=0.0):
        """
        Generates text from a conversation "messages" 
        which is a list of dicts like:
         [
           {"role": "system", "content": "..."},
           {"role": "user", "content": "..."}
         ]
        """
        pass
    
    def generate_from_prompt(self, prompt, max_tokens=2000, temperature=0.0):
        """Generate text from a single prompt string (for backward compatibility)"""
        messages = [{"role": "user", "content": prompt}]
        return self.generate(messages, max_tokens, temperature)

class OpenAIClient(LLMClient):
    """Client for OpenAI API"""
    
    def __init__(self, model="gpt-4", api_key=None):
        try:
            import openai
            self.openai = openai
            self.model = model
            if api_key:
                self.openai.api_key = api_key
            else:
                # Try to get from environment
                self.openai.api_key = os.environ.get("OPENAI_API_KEY")
                if not self.openai.api_key:
                    raise ValueError("OpenAI API key not found. Please provide it or set the OPENAI_API_KEY environment variable.")
        except ImportError:
            raise ImportError("openai package not found. Please install it with 'pip install openai'.")
        self.use_max_completion = self.model.startswith("o")
    
    
    def generate(self, messages, max_tokens=2000, temperature=0.0):
        """Generate text using OpenAI API"""
        try:
            # Build args dynamically based on model
            args = {
                "model": self.model,
                "messages": messages
            }

            # If it's an 'o1' model, use 'max_completion_tokens'
            if self.use_max_completion:
                args["max_completion_tokens"] = max_tokens
            else:
                args["max_tokens"] = max_tokens

            # Only pass temperature if the model supports it
            if not self.use_max_completion:
                args["temperature"] = temperature

            response = self.openai.chat.completions.create(**args)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return None

class OllamaClient(LLMClient):
    """Client for Ollama API"""
    
    def __init__(self, model="llama3", api_url="http://localhost:11434"):
        self.model = model
        self.api_url = api_url
    
    def check_connection(self):
        """Check if we can connect to the Ollama server."""
        try:
            import requests
            # Try to access the Ollama API
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def generate(self, messages, max_tokens=20000, temperature=0.0):
        """Generate text using Ollama API's chat endpoint"""
        try:
            # Use the /api/chat endpoint
            response = requests.post(
                f"{self.api_url}/api/chat", # Changed endpoint
                json={
                    "model": self.model,
                    "messages": messages, # Pass the messages list directly
                    "options": { # Ollama uses an 'options' object for parameters
                        "num_predict": max_tokens, # Parameter name might differ slightly based on Ollama version
                        "temperature": temperature
                    },
                    "stream": False
                },
                timeout=300
            )
            if response.status_code == 200:
                try:
                    # Parse the chat response structure
                    response_data = response.json()
                    # The response content is usually in response_data['message']['content']
                    return response_data.get("message", {}).get("content", "")
                except json.JSONDecodeError:
                    # Handle cases where response is not valid JSON
                    print(f"Warning: Ollama response was not valid JSON: {response.text}")
                    return response.text
            else:
                print(f"Error with Ollama API: {response.status_code}, {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama API: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred with Ollama API: {e}")
            return None