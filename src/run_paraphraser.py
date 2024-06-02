"""This module will run the paraphrase library."""

import time
from typing import Any

import torch
from absl import app, flags, logging
from torch.utils.data import DataLoader

from src.general_utils import DictDataset
from src.paraphraser import Paraphraser

FLAGS = flags.FLAGS


flags.DEFINE_float("paraphrase_learning_rate", 0.00005, "The learning rate used to train the paraphrase model", lower_bound=0.0)
flags.DEFINE_string("para_model_path", "/tmp", "The main directory to save or load the paraphrase model from.")
flags.DEFINE_string("para_checkpoint_name", "last", "The checkpoint name to load the paraphrase from.")
flags.DEFINE_integer("no_repeat_ngram_size", 2, "Related to generation with beam search.")
flags.DEFINE_integer("paraphrase_generation_max_length", 1024, "Maximum length to use for paraphrase generation.")
flags.DEFINE_float("paraphrase_top_p", 0.99, "The top_p value used in nucleus sampling.")
flags.DEFINE_float("repetition_penalty", 10.0, "The penalty for repeating sequences in the diverse beam search algorithm.")
flags.DEFINE_float("diversity_penalty", 3.0, "The diversity penalty used in the diverse beam search algorithm.")
flags.DEFINE_float("diverse_beam_temperature", 0.7, "The temperature value used in diverse beam search.")
flags.DEFINE_integer("paraphrase_cache_capacity", 100000, "The maximum capacity of the cache.")


def example_test_train_loop(model: Paraphraser) -> None:
    """Do a complete test of the model."""
    text = ["Today seems to be a rainy day in Toronto, and I like it!", "I hate you bro."]
    data = model.prepare_text_for_generation(text)
    dataloader = DataLoader(DictDataset(data), batch_size=len(text), shuffle=False)
    start_time = time.time()
    for data in dataloader:
        logging.info(data)

        logging.info("diverse beam search testing.")
        paraphrases, log_ps = model.generate_paraphrases(data, num_return_seq=2, decoding_technique="diverse_beam_search")
        logging.info(paraphrases)
        logging.info(log_ps)

        logging.info("top_p testing.")
        paraphrases, log_ps = model.generate_paraphrases(data, num_return_seq=2, decoding_technique="top_p")
        logging.info(paraphrases)
        logging.info(log_ps)

    end_time = time.time()
    logging.info(f"Time took: {end_time-start_time}")

    # Test the training loop.
    output_text = ["Today seems to be a rainy day in Toronto", "I hate you!"]
    data = model.prepare_text_for_training(text, output_text)
    dataloader = DataLoader(DictDataset(data), batch_size=len(text), shuffle=True)
    epochs = 10
    for e in range(epochs):
        for data in dataloader:
            log_ps = model.paraphrase_forward_pass(data, train=True)
            loss = -torch.mean(log_ps, dim=0)
            model.optimizer.zero_grad()
            loss.backward()
            logging.info(f"epoch_{e} loss_value:{loss.item()}")
            model.optimizer.step()
            model.scheduler.step(e)

    logging.info("Testing caching latency with paraphrases.")
    start_time = time.time()
    for data in dataloader:
        logging.info("diverse beam search testing.")
        paraphrases, log_ps = model.generate_paraphrases(data, num_return_seq=2, decoding_technique="diverse_beam_search")
        logging.info(paraphrases)
        logging.info(log_ps)
    end_time = time.time()
    logging.info(f"Time took without cache: {end_time-start_time}")

    start_time = time.time()
    for data in dataloader:
        logging.info("diverse beam search testing.")
        paraphrases, log_ps = model.generate_paraphrases(
            data, num_return_seq=2, decoding_technique="diverse_beam_search", use_internal_cache=True
        )
        logging.info(paraphrases)
        logging.info(log_ps)
    end_time = time.time()
    logging.info(f"Time took while cache building: {end_time-start_time}")

    logging.info(f"last_lr before saving: {model.scheduler.get_last_lr()}")
    model.save_to_checkpoint("/tmp", "testing_stage")

    # Create a different model and load it.
    FLAGS.para_model_path = "/tmp"
    FLAGS.para_checkpoint_name = "testing_stage"
    new_model = Paraphraser(device=model.device)
    new_model.load_from_checkpoint(FLAGS.para_model_path, FLAGS.para_checkpoint_name)
    new_model.to_device()
    logging.info(new_model.scheduler.get_last_lr())

    for data in dataloader:
        logging.info("top_p testing.")
        paraphrases, log_ps = new_model.generate_paraphrases(data, num_return_seq=2, decoding_technique="top_p")
        logging.info(paraphrases)
        logging.info(log_ps)

    for p1, p2 in zip(model.model.parameters(), new_model.model.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            logging.info(False)
            raise Exception("Not same models after saving and loading!")

    logging.info("Same models after saving and loading!")

    for e in range(10):
        for data in dataloader:
            log_ps = new_model.paraphrase_forward_pass(data, train=True)
            loss = -torch.mean(log_ps, dim=0)
            new_model.optimizer.zero_grad()
            loss.backward()
            logging.info(f"epoch_{e} loss_value:{loss.item()}")
            new_model.optimizer.step()
            new_model.scheduler.step(e + epochs)

    logging.info("Testing caching with paraphrases.")
    dataloader = DataLoader(DictDataset(data), batch_size=len(text), shuffle=False)
    start_time = time.time()
    for data in dataloader:
        logging.info("diverse_beam_search testing.")
        cached_paraphrases, cached_log_ps = new_model.generate_paraphrases(
            data, num_return_seq=2, decoding_technique="diverse_beam_search", use_internal_cache=True
        )
        logging.info(cached_paraphrases)
        logging.info(cached_log_ps)
    end_time = time.time()
    logging.info(f"Time took after cache: {end_time-start_time}")

    start_time = time.time()
    for data in dataloader:
        logging.info("diverse_beam_search testing.")
        paraphrases, log_ps = new_model.generate_paraphrases(
            data, num_return_seq=2, decoding_technique="diverse_beam_search", use_internal_cache=False
        )
        logging.info(paraphrases)
        logging.info(log_ps)
    end_time = time.time()
    logging.info(f"Time took without cache: {end_time-start_time}")

    assert paraphrases == cached_paraphrases
    assert torch.all(log_ps == cached_log_ps)

    start_time = time.time()
    for i in range(10):
        for data in dataloader:
            logging.info("mixed testing.")
            paraphrases, log_ps = new_model.generate_paraphrases(
                data, num_return_seq=2, decoding_technique="mixed", use_internal_cache=False
            )
            logging.info(paraphrases)
            logging.info(log_ps)

            logging.info("top p testing.")
            paraphrases, log_ps = new_model.generate_paraphrases(
                data, num_return_seq=2, decoding_technique="top_p", use_internal_cache=False
            )
            logging.info(paraphrases)
            logging.info(log_ps)

            logging.info("diverse beam search testing.")
            paraphrases, log_ps = new_model.generate_paraphrases(
                data, num_return_seq=2, decoding_technique="diverse_beam_search", use_internal_cache=False
            )
            logging.info(paraphrases)
            logging.info(log_ps)
    end_time = time.time()
    logging.info(f"Time took for 10 prediction loops: {end_time-start_time}")


def main(argv: Any) -> None:
    """Example function to launch the train and generate functions."""
    del argv

    logging.info("Testing the model on gpu!")
    model = Paraphraser(
        device="cuda:0",
        seed=42,
        para_checkpoint_name=FLAGS.para_checkpoint_name,
        para_model_path=FLAGS.para_model_path,
        paraphrase_cache_capacity=FLAGS.paraphrase_cache_capacity,
        diverse_beam_temperature=FLAGS.diverse_beam_temperature,
        repetition_penalty=FLAGS.repetition_penalty,
        no_repeat_ngram_size=FLAGS.no_repeat_ngram_size,
        paraphrase_generation_max_length=FLAGS.paraphrase_generation_max_length,
        paraphrase_learning_rate=FLAGS.paraphrase_learning_rate,
        paraphrase_top_p=FLAGS.paraphrase_top_p,
    )
    model.to_device()
    example_test_train_loop(model)
    del model


if __name__ == "__main__":
    app.run(main)
