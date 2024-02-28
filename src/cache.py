"""LRU Cache implementation with save and load functionality to disk."""

from collections import OrderedDict
from typing import Any, Generic, Hashable, Optional, TypeVar

import torch
from absl import app, logging

T = TypeVar("T")


class LruCache(Generic[T]):
    def __init__(self, capacity: int, filename: Optional[str] = None) -> None:
        """Define the capacity of the cache."""
        self.capacity = capacity
        self.cache: OrderedDict[Hashable, T] = OrderedDict()
        self.filename = filename

    def get(self, key: Hashable) -> Optional[T]:
        """Return the element."""
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def insert(self, key: Hashable, value: T) -> None:
        """Insert the element."""
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __len__(self) -> int:
        """Return the length of the cache."""
        return len(self.cache)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()

    def save(self) -> None:
        """Saves the in-memory cache contents pickle file.

        Raises:
            ValueError: If filename is not set.
        """
        if not self.filename:
            raise ValueError("Filename not set for saving.")
        torch.save(self.cache, self.filename)
        logging.info(f"Cache data saved to {self.filename}")

    def load(self) -> None:
        """Loads the cache contents from a pickle file.

        Raises:
            ValueError: If filename is not set.
            FileNotFoundError: If the file does not exist.
        """
        if not self.filename:
            raise ValueError("Filename not set for loading.")
        try:
            self.cache = torch.load(self.filename, map_location="cpu")
        except FileNotFoundError:
            logging.info(f"File not found {self.filename}")
        logging.info(f"Cache data loaded from {self.filename}")

    def load_to_device(self, device: str) -> None:
        """Regenerate the cache data and move all the tensors to the device."""
        for key, value in self.cache.items():
            paraphrases, log_ps = value
            if isinstance(log_ps, torch.Tensor):
                self.cache[key] = (paraphrases, log_ps.to(device))


def main(argv: Any) -> None:
    """Example function to test the cache."""
    del argv

    logging.info("Testing the in-memory cache!")
    cache: LruCache = LruCache(capacity=2, filename="/tmp/my_cache.bin")
    cache.insert(key="My first key", value=["value A", "value B"])
    cache.insert(key="My second key", value=["value C", "value D"])
    cache.insert(key="My third key", value=["value E", "value F"])
    cache.insert(key="My third key", value=["value G", "value H"])

    logging.info(cache.cache)
    cache.save()

    new_cache: LruCache = LruCache(capacity=2, filename="/tmp/my_cache.bin")
    new_cache.load()
    logging.info(new_cache.cache)
    logging.info(new_cache.get("My third key"))
    assert new_cache.get("My third key") == ["value G", "value H"]
    assert new_cache.get("My second key") == ["value C", "value D"]
    new_cache.clear()
    cache.clear()
    del new_cache
    del cache


if __name__ == "__main__":
    app.run(main)
