import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import torch
from absl import app, flags
from bitsandbytes.optim.adamw import PagedAdamW8bit
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.base_lm import BaseLM
from src.galore_torch import GaLoreAdamW8bit
from src.general_utils import DictDataset, test_loop, train_loop, white_space_fix
from src.model_utils import clear_cache, llama2_log_of_labels, lm_logits, mlm_log_of_labels, set_random_seed
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("output_file", "llama3_8b_instruction_10_shot.predicted.tsv", "the name of file to read data to.")

train_batch_size = 4
eval_batch_size = 8
lm_input_max_length = 3000 - 32
lm_output_max_length = 32
lm_top_p = 0.9
temperature = 0.6
learning_rate = 0.00005

# folder to store models and predictions.
model_path = "/scratch/ssd004/scratch/snajafi/checkpoints/llama3-squadv2.0"

# related to lora
r = 16
lora_alpha = 8
lora_dropout = 0.05

# create the in-context input for llama.
# Example context, question from squad.
contexts = [
    """Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.""",
    """Estonian belongs to the Finnic branch of the Uralic languages, along with Finnish, Karelian, and other nearby languages. The Uralic languages do not belong to the Indo-European languages. Estonian is distantly related to Hungarian and to the Sami languages.""",
    """Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".""",
    """The ISO 216 system used in most other countries is based on the surface area of a sheet of paper, not on a sheet's width and length. It was first adopted in Germany in 1922 and generally spread as nations adopted the metric system. The largest standard size paper is A0 (A zero), measuring one square meter (approx. 1189 × 841 mm). Two sheets of A1, placed upright side by side fit exactly into one sheet of A0 laid on its side. Similarly, two sheets of A2 fit into one sheet of A1 and so forth. Common sizes used in the office and the home are A4 and A3 (A3 is the size of two A4 sheets).""",
    """During the rule of the succeeding Hanoverian dynasty, power was gradually exercised more by parliament and the government. The first Hanoverian monarch, George I, relied on his ministers to a greater extent than did previous monarchs. Later Hanoverian monarchs attempted to restore royal control over legislation: George III and George IV both openly opposed Catholic Emancipation and asserted that to grant assent to a Catholic emancipation bill would violate the Coronation Oath, which required the sovereign to preserve and protect the established Church of England from Papal domination and would grant rights to individuals who were in league with a foreign power which did not recognise their legitimacy. However, George IV reluctantly granted his assent upon the advice of his ministers. Thus, as the concept of ministerial responsibility has evolved, the power to withhold royal assent has fallen into disuse, both in the United Kingdom and in the other Commonwealth realms.""",
    '''Chopin's successes as a composer and performer opened the door to western Europe for him, and on 2 November 1830, he set out, in the words of Zdzisław Jachimecki, "into the wide world, with no very clearly defined aim, forever." With Woyciechowski, he headed for Austria, intending to go on to Italy. Later that month, in Warsaw, the November 1830 Uprising broke out, and Woyciechowski returned to Poland to enlist. Chopin, now alone in Vienna, was nostalgic for his homeland, and wrote to a friend, "I curse the moment of my departure." When in September 1831 he learned, while travelling from Vienna to Paris, that the uprising had been crushed, he expressed his anguish in the pages of his private journal: "Oh God! ... You are there, and yet you do not take vengeance!" Jachimecki ascribes to these events the composer's maturing "into an inspired national bard who intuited the past, present and future of his native Poland."''',
    """Each of these four dialects was associated with an independent kingdom on the island. Of these, Northumbria south of the Tyne, and most of Mercia, were overrun by the Vikings during the 9th century. The portion of Mercia that was successfully defended, and all of Kent, were then integrated into Wessex under Alfred the Great. From that time on, the West Saxon dialect (then in the form now known as Early West Saxon) became standardised as the language of government, and as the basis for the many works of literature and religious materials produced or translated from Latin in that period.""",
    """Exposure to antibiotics early in life is associated with increased body mass in humans and mouse models. Early life is a critical period for the establishment of the intestinal microbiota and for metabolic development. Mice exposed to subtherapeutic antibiotic treatment (STAT)– with either penicillin, vancomycin, penicillin and vancomycin, or chlortetracycline had altered composition of the gut microbiota as well as its metabolic capabilities. Moreover, research have shown that mice given low-dose penicillin (1 μg/g body weight) around birth and throughout the weaning process had an increased body mass and fat mass, accelerated growth, and increased hepatic expression of genes involved in adipogenesis, compared to controlled mice. In addition, penicillin in combination with a high-fat diet increased fasting insulin levels in mice. However, it is unclear whether or not antibiotics cause obesity in humans. Studies have found a correlation between early exposure of antibiotics (<6 months) and increased body mass (at 10 and 20 months). Another study found that the type of antibiotic exposure was also significant with the highest risk of being overweight in those given macrolides compared to penicillin and cephalosporin. Therefore, there is correlation between antibiotic exposure in early life and obesity in humans, but whether or not there is a causal relationship remains unclear. Although there is a correlation between antibiotic use in early life and obesity, the effect of antibiotics on obesity in humans needs to be weighed against the beneficial effects of clinically indicated treatment with antibiotics in infancy.""",
    """The term "matter" is used throughout physics in a bewildering variety of contexts: for example, one refers to "condensed matter physics", "elementary matter", "partonic" matter, "dark" matter, "anti"-matter, "strange" matter, and "nuclear" matter. In discussions of matter and antimatter, normal matter has been referred to by Alfvén as koinomatter (Gk. common matter). It is fair to say that in physics, there is no broad consensus as to a general definition of matter, and the term "matter" usually is used in conjunction with a specifying modifier.""",
    """Database transactions can be used to introduce some level of fault tolerance and data integrity after recovery from a crash. A database transaction is a unit of work, typically encapsulating a number of operations over a database (e.g., reading a database object, writing, acquiring lock, etc.), an abstraction supported in database and also other systems. Each transaction has well defined boundaries in terms of which program/code executions are included in that transaction (determined by the transaction's programmer via special transaction commands).""",
]

questions = [
    "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
    "What Uralic language branch does not contain Estonian?",
    "When did Beyonce start becoming popular?",
    "When was the ISO 1189 system first adopted in Germany?",
    "Which monarch relied on his ministers more than any of his predecessors?",
    "What historian commented that the events involving Frédéric's friend in Poland contributed to his maturing?",
    "who over ran most of Mercia in the 900's?",
    "What does STAT stand for?",
    "Physics has broadly agreed on the definition of what?",
    "What is a unit of play called in a database?",
]

gold_answers = [
    "Saint Bernadette Soubirous",
    "<no_answer>",
    "in the late 1990s",
    "<no_answer>",
    "George I",
    "Zdzisław Jachimecki",
    "<no_answer>",
    "subtherapeutic antibiotic treatment",
    "<no_answer>",
    "<no_answer>",
]

instruction = """This task is about writing a correct answer for the reading comprehension task. Based on the information provided in a given passage, you should identify the shortest continuous text span from the passage that serves as an answer to the given question. Avoid answers that are incorrect or provides incomplete justification for the question. Do not generate the explanations for your answer. If you cannot find the answer from the passage for the given question, then generate the <no_answer> tag in the output."""

# create chat template for llama3.
instruction_llama = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"

icl_input = f"{instruction}\n\n(Examples)"
for idx, context_example in enumerate(contexts):
    icl_input = f"{icl_input}\n\nPassage_{idx+1}: {context_example}\nQuestion_{idx+1}: {questions[idx]}\nAnswer_{idx+1}: {gold_answers[idx]}"


instruction_llama = instruction_llama.format(instruction=icl_input)
input_example_template = (
    "<|start_header_id|>user<|end_header_id|>\n\nPassage_{idx}: {passage}\nQuestion_{idx}: {question}<|eot_id|>"
)
"""Load LM efficiently."""

# Make sure we have some tokens defined for the LM, if not defined in the model.

# Specific for Llama3
_EXTRA_TOKENS = {
    "pad_token": "<|reserved_special_token_0|>",
}

target_modules = ["q_proj", "v_proj", "o_proj", "k_proj"]


def load_peft_model(
    model: PreTrainedModel,
    adapter_name: str = "lora",
    is_trainable: bool = False,
    model_type: str = "causal_lm",
    lora_target_modules: List[str] = target_modules,
) -> torch.nn.Module:
    """Load a trained PEFT adapter to the base model and return the PeftModel.

    Args:
    ----
        model: the main model.
        num_quantized_bits: number of bits in the loaded model.
        adapter_name: e.g. lora.
        is_trainable: train or inference mode.
        model_type: causal lm or seq-to-seq.
        lora_target_modules: which modules to train with lora.

    Returns:
    -------
        The PEFT model and tokenizer.
    """
    if model_type == "causal_lm":
        task_type = TaskType.CAUSAL_LM
    elif model_type == "seq_to_seq_lm":
        task_type = TaskType.SEQ_2_SEQ_LM

    if adapter_name == "lora":
        peft_config = LoraConfig(
            task_type=task_type,
            inference_mode=not is_trainable,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            init_lora_weights=True,
            target_modules=lora_target_modules,
        )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_model_and_tokenizer(
    model_id: str, model_type: str, model_dtype: torch.dtype, attn_implementation: str, load_in_4bit: Optional[bool] = True
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the model and tokenizer.

    Args:
    ----
        model_id: the id for the pre-trained model.
        model_type: causal lm or seq_to_seq_lm.
        model_dtype: model data type.
        load_in_4bit: Whether to load in 4 bit quantization.

    Returns:
    -------
        The model and tokenizer.
    """
    # load model
    if model_type == "causal_lm":
        ModelClass = AutoModelForCausalLM
    elif model_type == "seq_to_seq_lm":
        ModelClass = AutoModelForSeq2SeqLM
    model_args: Dict[str, Any] = {"use_cache": False, "attn_implementation": attn_implementation, "torch_dtype": model_dtype}
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_args["torch_dtype"],
            bnb_4bit_use_double_quant=True,
        )
        model_args["quantization_config"] = quant_config
    model = ModelClass.from_pretrained(
        model_id,
        **model_args,
    )

    # load tokenizer
    # padding is from left for the decoder only models.
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.add_special_tokens(_EXTRA_TOKENS)

    if torch.cuda.is_available():
        # extend embeddings to a multiple so we use Tensor cores
        multiple = 64 if "A100" in torch.cuda.get_device_name() else 8
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=multiple)
    else:
        raise Exception("No CUDA Found!")

    # re-define token ids for the model.
    for extra_token_key, extra_token_val in _EXTRA_TOKENS.items():
        extra_token_id = tokenizer.convert_tokens_to_ids([extra_token_val])[0]
        model.config.__setattr__(f"{extra_token_key}_id", extra_token_id)
        model.generation_config.__setattr__(f"{extra_token_key}_id", extra_token_id)

    return model, tokenizer


class LlamaQA(BaseLM):
    """Class to implement Llama for QA task."""

    def __init__(
        self,
        mode: str,
        device: str,
        seed: int = 42,
    ) -> None:
        super().__init__(device, "main_lm", seed)
        self.device = device
        model, tokenizer = load_model_and_tokenizer(
            model_id="/model-weights/Meta-Llama-3-8B-Instruct",
            model_type="causal_lm",
            model_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            load_in_4bit=False,
        )
        self.model = model
        self.tokenizer = tokenizer
        """# to train the main lm, we update all of its parameters.

        galore_params = [] target_modules_list = ["attn", "mlp"] for
        module_name, module in self.model.named_modules():     if not
        isinstance(module, torch.nn.Linear):         continue     if not
        any(target_key in module_name for target_key in
        target_modules_list):         continue     print('enable GaLore
        for weights in module: ', module_name)
        galore_params.append(module.weight) id_galore_params = [id(p)
        for p in galore_params] # make parameters without "rank" to
        another group regular_params = [p for p in
        self.model.parameters() if id(p) not in id_galore_params] # then
        call galore_adamw param_groups = [{'params': regular_params},
        {'params': galore_params, 'rank': 128, 'update_proj_gap': 16,
        'scale': 0.25, 'proj_type': 'std'}] self.optimizer =
        GaLoreAdamW8bit(param_groups, lr=learning_rate)
        """
        self.optimizer = PagedAdamW8bit(self.model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, eta_min=learning_rate / 5.0)

        # required for llama3.
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    def train(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Using the Llama, run a forward computation over the batch, compute
        the log probability over the batch.

        This will be used for training.
        """
        self.train_mode_on()
        loaded_batch = self.data_to_device(
            batch, keys=["lm_input_ids_for_train", "lm_attention_mask_for_train", "lm_attention_mask_for_generation"]
        )
        input_ids = loaded_batch["lm_input_ids_for_train"]
        attention_mask = loaded_batch["lm_attention_mask_for_train"]
        original_len_without_answer = torch.sum(loaded_batch["lm_attention_mask_for_generation"], dim=1)
        with torch.set_grad_enabled(True):
            logits = lm_logits(
                model=self.model,
                input_ids=input_ids,
                input_mask=attention_mask,
            )
            batch_size, seq_len = input_ids.size()
            masked_labels = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
            prompt_mask = torch.arange(seq_len, device=self.device).expand(
                batch_size, seq_len
            ) < original_len_without_answer.unsqueeze(1)
            masked_labels = masked_labels.masked_fill(prompt_mask == 1, -100)
            return llama2_log_of_labels(logits=logits, labels=masked_labels, loss_func=self.loss_func)

    def generation_pass(self, batch: torch.utils.data.Dataset) -> Tuple[List[str], torch.Tensor]:
        """Using the Llamma, generate new text.

        This will be used for inference.
        """
        self.predict_mode_on()
        loaded_batch = self.data_to_device(batch, keys=["lm_input_ids_for_generation", "lm_attention_mask_for_generation"])
        input_ids = loaded_batch["lm_input_ids_for_generation"]
        attention_mask = loaded_batch["lm_attention_mask_for_generation"]
        with torch.no_grad():
            # more look here:
            # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L130
            predictions_output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                top_p=lm_top_p,
                temperature=temperature,
                max_length=lm_input_max_length + lm_output_max_length,
                num_return_sequences=1,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=True,
                renormalize_logits=True,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = input_ids.size()[1]
        selected_samples = predictions_output.sequences[:, prompt_len:]
        predictions_str = self.tokenizer.batch_decode(selected_samples, skip_special_tokens=True)
        logits_list = list(predictions_output.logits)
        logits = torch.stack(logits_list, dim=1)
        labels_to_consider = selected_samples.masked_fill(selected_samples == self.tokenizer.pad_token_id, -100)
        final_log_ps = mlm_log_of_labels(logits=logits, labels=labels_to_consider, loss_func=self.loss_func)
        actual_lens = torch.sum(torch.where(labels_to_consider > 0, 1, 0), dim=1)
        # Average log probs per token (length normalization).
        return predictions_str, final_log_ps / actual_lens

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop."""
        answers, log_ps = self.generation_pass(batch)
        log_ps = log_ps.cpu().detach().numpy()
        for idx, answer in enumerate(answers):
            output_row = {
                "potential_answer": answer,
                "prediction_score": log_ps[idx],
                "id": batch["ids"][idx]
            }
            yield output_row


def prepare_text(texts: List[str], ids: List[str], model: PreTrainedModel) -> Dict[str, Any]:
    """Convert texts to ids and return the dataset required for training and
    inference."""
    input_encodings_for_generation = model.tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=lm_input_max_length,
        add_special_tokens=False,
    )
    data = {
        "ids": ids,
        "lm_input_ids_for_generation": input_encodings_for_generation.input_ids,
        "lm_attention_mask_for_generation": input_encodings_for_generation.attention_mask,
    }
    return data


def main(argv: Any) -> None:
    del argv

    # Create model.
    set_random_seed(42)
    model = LlamaQA(mode="test", device="cuda:0", seed=42)
    model.to_device()

    dataset = load_dataset("rajpurkar/squad_v2", split="validation")

    next_example_number = len(contexts) + 1
    squad_inputs = []
    squad_ids = []
    for idx, row in enumerate(dataset):
        context = row["context"]
        question = row["question"]
        user_final_message = input_example_template.format(idx=next_example_number, passage=context, question=question)
        squad_input = f"{instruction_llama}{user_final_message}"
        squad_inputs.append(squad_input)
        squad_ids.append(row["id"])

    data = prepare_text(squad_inputs, squad_ids, model)
    dataset = DictDataset(data)
    data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

    # Run on the Test Data.
    test_loop(
        model=model,
        mode="test",
        model_path=model_path,
        prediction_file_name=FLAGS.output_file,
        test_dataloader=data_loader,
        metric=None,
    )


if __name__ == "__main__":
    app.run(main)
