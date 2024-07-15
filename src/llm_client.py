"""Submit a parallel requests to the server.

Usage:
python3 src/llm_client.py --server_url "http://localhost:8080/v1" --request_batch_size 128 --num_threads 8 >> file.txt 2>&1
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

from absl import app, flags, logging
from numpy import mean
from openai import APIConnectionError, APIError, OpenAI, RateLimitError, Timeout
from openai.types import CompletionChoice

FLAGS = flags.FLAGS

flags.DEFINE_string("server_url", "http://gpuXXX:XXXX/v1", "address for the vllm server.")
flags.DEFINE_string("model_name", "/model-weights/Meta-Llama-3-8B-Instruct", "corresponding name in the server.")
flags.DEFINE_integer("max_new_tokens", 256, "max number of tokens to generate per sequence.")
flags.DEFINE_integer("max_retries", 5, "number of retries before failure.")
flags.DEFINE_integer("seconds_between_retries", 10, "sleep time between retries.")
flags.DEFINE_integer("request_batch_size", 128, "batch size to send group requests in one call.")
flags.DEFINE_integer("num_threads", 8, "number of threads for parallel client calls.")
flags.DEFINE_list("stop_token_ids", "128001,128009", "stop token ids for a particular model.")


class MyOpenAIClient:
    """A wrapper for sending requests for openai compatible server."""

    def __init__(
        self,
        server_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        max_new_tokens: int = 256,
        max_retries: int = 5,
        seconds_between_retries: int = 10,
        request_batch_size: int = 8,
        stop_token_ids: List[str] = ["128001", "128009"],
    ) -> None:
        """Initialize the OpenAI client."""

        self.server_url = server_url
        self.model_name = model_name
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        assert max_retries > 0
        assert seconds_between_retries > 0
        self.max_retries = max_retries
        self.seconds_between_retries = seconds_between_retries
        self.request_batch_size = request_batch_size
        self.stop_token_ids = [int(id) for id in stop_token_ids]
        self.client = OpenAI(base_url=self.server_url, api_key=self.api_key).with_options(max_retries=self.max_retries)

    def __call__(
        self,
        inputs: List[str],
        top_p: float = 0.9,
        temperature: float = 0.001,
        logprobs: bool = True,
        seed: int = 42,
    ) -> List[CompletionChoice]:
        """Send the request and get the output."""
        kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
            "max_tokens": self.max_new_tokens,
            "logprobs": logprobs,
            "seed": seed,
            "extra_body": {"stop_token_ids": self.stop_token_ids},
        }
        responses: List[CompletionChoice] = []
        num_chunks = math.ceil(len(inputs) / self.request_batch_size)
        for chunk_i in range(num_chunks):
            if (chunk_i + 1) * self.request_batch_size <= len(inputs):
                sub_inputs = inputs[chunk_i * self.request_batch_size : (chunk_i + 1) * self.request_batch_size]
            else:
                sub_inputs = inputs[chunk_i * self.request_batch_size :]
            self.request_counter = self.max_retries
            sub_responses = self.api_request(
                sub_inputs,
                **kwargs,
            )
            if sub_responses is not None:
                responses.extend(sub_responses)
            else:
                responses.extend([CompletionChoice(text="<API has failed!>")] * len(sub_inputs))
        return responses

    def api_request(self, inputs: List[str], **kwargs: Any) -> None | List[CompletionChoice]:
        """Send the actual request, deduct the counter for retrying."""
        self.request_counter -= 1
        try:
            responses = self.client.completions.create(model=self.model_name, prompt=inputs, stream=False, **kwargs)
            return responses.choices
        except (
            RateLimitError,
            APIConnectionError,
            APIError,
            Timeout,
        ) as e:
            if self.request_counter > 0:
                logging.info(f"Error: {e}. Waiting {self.seconds_between_retries} seconds before retrying.")
                time.sleep(self.seconds_between_retries)
                return self.api_request(inputs, **kwargs)
            else:
                logging.info(f"Final retry ended with error: {e}.")
                return None


def parallel_generator(
    server_url: str,
    model_name: str,
    inputs: List[str],
    num_threads: int = 2,
    max_new_tokens: int = 256,
    max_retries: int = 5,
    seconds_between_retries: int = 10,
    request_batch_size: int = 8,
    stop_token_ids: List[str] = ["128001", "128009"],
    top_p: float = 0.9,
    temperature: float = 0.001,
    logprobs: bool = True,
    seed: int = 42,
) -> List[CompletionChoice]:
    """Call the openai client in parallel using different threads."""
    final_responses = []
    assert num_threads > 0
    chunk_size = len(inputs) // num_threads
    if chunk_size == 0:
        num_threads = len(inputs)
        chunk_size = 1
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for thread_i in range(num_threads):
            if thread_i < num_threads - 1:
                sub_inputs = inputs[thread_i * chunk_size : (thread_i + 1) * chunk_size]
            else:
                sub_inputs = inputs[thread_i * chunk_size :]

            thread_client = MyOpenAIClient(
                server_url=server_url,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
                seconds_between_retries=seconds_between_retries,
                request_batch_size=request_batch_size,
                stop_token_ids=stop_token_ids,
            )
            future = executor.submit(thread_client, sub_inputs, top_p, temperature, logprobs, seed)
            futures.append(future)

        # Collect the results from each future.
        for future in futures:
            final_responses.extend(future.result())

    return final_responses


def main(argv: Any) -> None:
    """Example to use the client."""
    del argv

    sample_inputs = [
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

    instruction_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction} <|eot_id|>"
    input_template = "<|start_header_id|>user<|end_header_id|>\n\n{input} <|eot_id|>"
    full_instruction = instruction_template.format(instruction="provide a short answer for the question.")
    full_sample_inputs = [full_instruction + input_template.format(input=sample) for sample in sample_inputs]

    start_time = time.perf_counter()
    responses = parallel_generator(
        server_url=FLAGS.server_url,
        model_name=FLAGS.model_name,
        inputs=full_sample_inputs * 10,
        num_threads=FLAGS.num_threads,
        max_new_tokens=FLAGS.max_new_tokens,
        max_retries=FLAGS.max_retries,
        seconds_between_retries=FLAGS.seconds_between_retries,
        request_batch_size=FLAGS.request_batch_size,
        stop_token_ids=FLAGS.stop_token_ids,
    )
    end_time = time.perf_counter()
    logging.info(len(sample_inputs))
    logging.info(len(responses))
    for response in responses:
        msg = f"Text: {response.text}, LogProbs: {mean(response.logprobs.token_logprobs)}"
        logging.info(msg)

    logging.info(f"Finished in {end_time - start_time} seconds!")


if __name__ == "__main__":
    app.run(main)
