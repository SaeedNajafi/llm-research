"""Submit parallel requests to the openai server."""

import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from absl import app, flags, logging
from numpy import mean
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

FLAGS = flags.FLAGS

flags.DEFINE_string("openai_key", "XXX", "key for the openai account.")
flags.DEFINE_string("model_name", "gpt-4o-mini", "corresponding name in the server.")
flags.DEFINE_integer("max_new_tokens", 256, "max number of tokens to generate per sequence.")
flags.DEFINE_integer("max_retries", 5, "number of retries before failure.")
flags.DEFINE_integer("seconds_between_retries", 10, "sleep time between retries.")
flags.DEFINE_integer("request_batch_size", 1, "batch size to send group requests in one call.")
flags.DEFINE_integer("num_threads", 1, "number of threads for parallel client calls.")
flags.DEFINE_list("stop_tokens", "</s>", "stop tokens.")


class MyOpenAIClient:
    """A wrapper for sending requests for openai compatible server."""

    def __init__(
        self,
        model_name: str,
        api_key: str = "EMPTY",
        max_new_tokens: int = 256,
        max_retries: int = 5,
        seconds_between_retries: int = 10,
        request_batch_size: int = 8,
        stop_tokens: List[str] = ["</s>"],
    ) -> None:
        """Initialize the OpenAI client."""

        self.model_name = model_name
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        assert max_retries > 0
        assert seconds_between_retries > 0
        self.max_retries = max_retries
        self.seconds_between_retries = seconds_between_retries
        self.request_batch_size = request_batch_size
        self.stop_tokens = stop_tokens
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", self.api_key)).with_options(max_retries=self.max_retries)

    def __call__(
        self,
        messages: List[Dict[str, str]],
        top_p: float = 0.9,
        temperature: float = 0.001,
        logprobs: bool = True,
        seed: int = 42,
    ) -> List[ChatCompletion]:
        """Send the request and get the output."""
        kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
            "max_tokens": self.max_new_tokens,
            "logprobs": logprobs,
            "seed": seed,
            "stop": self.stop_tokens,
        }
        responses: List[ChatCompletion] = []
        num_chunks = math.ceil(len(messages) / self.request_batch_size)
        for chunk_i in range(num_chunks):
            if (chunk_i + 1) * self.request_batch_size <= len(messages):
                sub_messages = messages[chunk_i * self.request_batch_size : (chunk_i + 1) * self.request_batch_size]
            else:
                sub_messages = messages[chunk_i * self.request_batch_size :]
            self.request_counter = self.max_retries
            sub_responses = self.api_request(
                sub_messages,
                **kwargs,
            )
            if sub_responses is not None:
                responses.extend(sub_responses)
            else:
                choices = [Choice(index=-1, logprobs=None, message=ChatCompletionMessage(content="<API has failed!>"))] * len(
                    sub_messages
                )
                responses.extend(ChatCompletion(choices=choices, id=-1, model=self.model_name, created=-1))
        return responses

    def api_request(self, messages: List[Dict[str, str]], **kwargs: Any) -> None | List[ChatCompletion]:
        """Send the actual request, deduct the counter for retrying."""
        self.request_counter -= 1
        try:
            responses = self.client.chat.completions.create(model=self.model_name, messages=messages, **kwargs)
            return responses.choices
        except Exception as e:
            if self.request_counter > 0:
                logging.info(f"Error: {e}. Waiting {self.seconds_between_retries} seconds before retrying.")
                time.sleep(self.seconds_between_retries + random.randint(1, 5))
                return self.api_request(messages, **kwargs)
            else:
                logging.info(f"Final retry ended with error: {e}.")
                return None


def parallel_generator(
    api_key: str,
    model_name: str,
    messages: List[Dict[str, str]],
    num_threads: int = 2,
    max_new_tokens: int = 256,
    max_retries: int = 5,
    seconds_between_retries: int = 10,
    request_batch_size: int = 8,
    stop_tokens: List[str] = ["</s>"],
    top_p: float = 0.9,
    temperature: float = 0.001,
    logprobs: bool = True,
    seed: int = 42,
) -> List[ChatCompletion]:
    """Call the openai client in parallel using different threads."""
    final_responses = []
    assert num_threads > 0
    chunk_size = len(messages) // num_threads
    if chunk_size == 0:
        num_threads = len(messages)
        chunk_size = 1
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for thread_i in range(num_threads):
            if thread_i < num_threads - 1:
                sub_inputs = messages[thread_i * chunk_size : (thread_i + 1) * chunk_size]
            else:
                sub_inputs = messages[thread_i * chunk_size :]

            thread_client = MyOpenAIClient(
                api_key=api_key,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
                seconds_between_retries=seconds_between_retries,
                request_batch_size=request_batch_size,
                stop_tokens=stop_tokens,
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

    instruction_template = "<s> Instruction:\n{instruction} </s>"
    input_template = "<s> Input:\n{input} </s>"
    full_instruction = instruction_template.format(instruction="provide a short answer for the question.")

    full_messages = []
    for sample in sample_inputs:
        full_messages.append({"role": "user", "content": full_instruction + input_template.format(input=sample)})

    start_time = time.perf_counter()
    responses = parallel_generator(
        api_key=FLAGS.openai_key,
        model_name=FLAGS.model_name,
        messages=full_messages,
        num_threads=FLAGS.num_threads,
        max_new_tokens=FLAGS.max_new_tokens,
        max_retries=FLAGS.max_retries,
        seconds_between_retries=FLAGS.seconds_between_retries,
        request_batch_size=FLAGS.request_batch_size,
        stop_tokens=FLAGS.stop_tokens,
    )
    end_time = time.perf_counter()
    logging.info(len(sample_inputs))
    logging.info(len(responses))
    for response in responses:
        logprobs = [log_content.logprob for log_content in response.logprobs.content if log_content.logprob != 0]
        msg = f"Text: {response.message.content}, LogProbs: {mean(logprobs)}"
        logging.info(msg)

    logging.info(f"Finished in {end_time - start_time} seconds!")


if __name__ == "__main__":
    app.run(main)
