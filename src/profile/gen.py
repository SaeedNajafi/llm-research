import argparse
import time
from typing import List

import requests

# Change the ENDPOINT and MODEL_PATH to match your setup
ENDPOINT = "http://ng31201:3456/v1"
MODEL_PATH = "/home/saeednjf/nearline/rrg-afyshe/pre-trained-models/gemma-2-27b-it"

# Configuration
API_KEY = "EMPTY"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Sample prompts for testing
PROMPTS = [
    "Translate the following English text to French: 'Hello, how are you?'",
    "What is the square root of 144?",
    "Summarize the following paragraph: 'Artificial intelligence refers to ...'",
    "Explain the process of photosynthesis in plants.",
    "What are the main differences between classical and quantum physics?",
    "Summarize the plot of 'To Kill a Mockingbird' by Harper Lee.",
    "Describe the economic impacts of climate change on agriculture.",
    "Translate the following sentence into Spanish: 'Where is the closest grocery store?'",
    "How does a lithium-ion battery work?",
    "Provide a brief biography of Marie Curie.",
    "What are the key factors that led to the end of the Cold War?",
    "Write a poem about the sunset over the ocean.",
    "Explain the rules of chess.",
    "What is blockchain technology and how does it work?",
    "Give a step-by-step guide on how to bake chocolate chip cookies.",
    "Describe the human digestive system.",
    "What is the theory of relativity?",
    "How to perform a basic oil change on a car.",
    "What are the symptoms and treatments for type 2 diabetes?",
    "Summarize the last episode of 'Game of Thrones'.",
    "Explain the role of the United Nations in world peace.",
    "Describe the culture and traditions of Japan.",
    "Provide a detailed explanation of the stock market.",
    "How do solar panels generate electricity?",
    "What is machine learning and how is it applied in daily life?",
    "Discuss the impact of the internet on modern education.",
    "Write a short story about a lost dog finding its way home.",
    "What are the benefits of meditation?",
    "Explain the process of recycling plastic.",
    "What is the significance of the Magna Carta?",
    "How does the human immune system fight viruses?",
    "Describe the stages of a frog's life cycle.",
    "Explain Newton's three laws of motion.",
    "What are the best practices for sustainable farming?",
    "Give a history of the Olympic Games.",
    "What are the causes and effects of global warming?",
    "Write an essay on the importance of voting.",
    "How is artificial intelligence used in healthcare?",
    "What is the function of the Federal Reserve?",
    "Describe the geography of South America.",
    "Explain how to set up a freshwater aquarium.",
    "What are the major works of William Shakespeare?",
    "How do antibiotics work against bacterial infections?",
    "Discuss the role of art in society.",
    "What are the main sources of renewable energy?",
    "How to prepare for a job interview.",
    "Describe the life cycle of a butterfly.",
    "What are the main components of a computer?",
    "Write a review of the latest Marvel movie.",
    "What are the ethical implications of cloning?",
    "Explain the significance of the Pyramids of Giza.",
    "Describe the process of making wine.",
    "How does the GPS system work?",
]


def send_request(prompts: List[str], endpoint: str, model_path: str) -> None | float:
    data = {"model": model_path, "prompt": prompts, "max_tokens": 100}
    start_time = time.time()
    response = requests.post(f"{endpoint}/completions", headers=HEADERS, json=data)
    duration = time.time() - start_time
    if response.status_code == 200:
        return duration
    else:
        return None


def main() -> None:
    # Create the parser
    parser = argparse.ArgumentParser(description="Send sample jobs to see if the server works.")

    # Add the arguments
    parser.add_argument("--endpoint", type=str, help="server address")
    parser.add_argument("--model_path", type=str, help="model path")

    # Execute the parse_args() method
    args = parser.parse_args()

    for i in range(10):
        print("Sending 20x requests 0-52...")
        send_request(PROMPTS * 20, args.endpoint, args.model_path)
    print("Done!")


if __name__ == "__main__":
    main()
