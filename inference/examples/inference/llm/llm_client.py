import time
from typing import Any, List

from openai import OpenAI
from openai.error import APIConnectionError, APIError, RateLimitError, ServiceUnavailableError, Timeout


class OpenAIChatClient:
    def __init__(
        self,
        server_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        max_new_tokens: int = 256,
        max_retries: int = 5,
        seconds_between_retries: int = 10,
    ) -> None:
        """Initialize the OpenAI chat complete client."""

        self.server_url = server_url
        self.model_name = model_name
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        assert self.max_retries > 0
        self.seconds_between_retries = seconds_between_retries
        self.client = OpenAI(base_url=server_url, api_key=self.api_key).with_options(max_retries=self.max_retries)

    def __call__(
        self,
        inputs: List[str],
        request_mode: str,
        candidates: int = 1,
        top_p: float = 0.95,
        temperature: float = 0.001,
        logprobs: bool = True,
        seed: int = 42,
    ) -> List[List[str]]:
        """Send the request and get the output."""
        responses = []
        for input in inputs:
            kwargs = {"temperature": temperature, "top_p": top_p, "n": candidates, "logprobs": logprobs, "seed": seed}

            kwargs["max_tokens"] = self.max_new_tokens
            self.request_counter = self.max_retries
            response = self.api_request(
                input,
                request_mode,
                **kwargs,
            )
            if candidates == 1:
                if len(response):
                    responses.append(response[0])
                else:
                    responses.append(["<request has failed!>"])
            else:
                if len(response):
                    responses.append(response)
                else:
                    responses.append(["<request has failed!>"] * candidates)
        return responses

    def api_request(self, input: str, request_mode: str, **kwargs: Any) -> None | List[str]:
        """Send the actual request, deduct the counter for retrying."""
        self.request_counter -= 1
        try:
            if request_mode == "chat":
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": input}],
                    **kwargs,
                )
                return [r.message.content for r in res.choices]
            elif request_mode == "completions":
                res = self.client.completions.create(model=self.model_name, prompt=input, **kwargs)
                return [r.text for r in res.choices]
        except (
            RateLimitError,
            APIConnectionError,
            ServiceUnavailableError,
            APIError,
            Timeout,
        ) as e:
            if self.request_counter > 0:
                print(f"Error: {e}. Waiting {self.seconds_between_retries} seconds before retrying.")
                time.sleep(self.seconds_between_retries)
                return self.api_request(input, request_mode, **kwargs)
            else:
                print(f"Final retry ended with error: {e}.")
                return None
