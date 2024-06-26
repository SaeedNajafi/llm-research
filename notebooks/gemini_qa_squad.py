import json
import time
from typing import Any

import google.generativeai as genai
from absl import app, flags
from datasets import load_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string("output_file", "a name", "the name of file to read data to.")
genai.configure(api_key="AIzaSyAn3lW6YIu4acri0Ydljo_306jAA-Cuao4")


def main(argv: Any) -> None:
    del argv
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)

    model = genai.GenerativeModel("models/gemini-1.0-pro-latest")

    # Generation Config
    generation_config = genai.types.GenerationConfig(candidate_count=1, stop_sequences=["</s>"], temperature=0.0)

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
        "Vikings",
        "subtherapeutic antibiotic treatment",
        "<no_answer>",
        "<no_answer>",
    ]

    # Prompt used with Gemini 1.5 pro to generate these explanations.
    # I will give you a passage, a question about the passage, and the final answer for that question. Your task is to generate explanations and a thought process which answers the question.
    # Passage: Question: Answer: Explanations and Thought Process: Let's think step by step.

    # I will give you a passage, a question that is not answerable from the information in the passage, and the <no_answer> tag. Your task is to generate explanations for the final <no_answer> tag.

    explanations = [
        """Identify the question's focus: The question asks specifically about the individual to whom the Virgin Mary appeared in Lourdes.
Find the relevant section: The passage mentions a replica of the grotto at Lourdes, France, where the Virgin Mary is said to have appeared. It also mentions that this appearance was to Saint Bernadette Soubirous in 1858.
Reasoning: Therefore, we know the Virgin Mary appeared at Lourdes to Saint Bernadette Soubirous.""",
        """Identify the question's focus: The question asks for a Uralic language branch that does NOT contain Estonian.
Find the relevant section: The passage states that Estonian belongs to the Finnic branch of the Uralic languages.
Reasoning: Therefore, any Uralic language branch that is NOT the Finnic branch would be a valid answer. However, the passage does not mention any other Uralic language branches besides the Finnic branch. We only know that Estonian is distantly related to Hungarian and the Sami languages, but we don't know if these are part of other Uralic branches. Thus, we lack the information to determine which Uralic language branches do not contain Estonian.""",
        """Identify the question's focus: The question asks about Beyonce's rise to fame.
Find the relevant section: The passage mentions her early career, saying she "performed in various singing and dancing competitions as a child". The passage then states "she rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child".
Reasoning: Therefore, the statement directly answers the question, stating that her rise to fame began in the late 1990s.""",
        """Identify the question's focus: The question asks about the adoption of the ISO 1189 system in Germany.
Find the relevant section: The passage mentions the ISO 216 system, not ISO 1189. The passage states that the ISO 216 system was first adopted in Germany in 1922.
Reasoning: Therefore, the question asks about a system not discussed in the passage, making it unanswerable.""",
        """Identify the question's focus: The question asks about which Hanoverian monarch relied most heavily on their ministers.
Find the relevant section: The passage mentions that Hanoverian monarchs gradually shifted power to Parliament and the government, and that George I relied on his ministers "to a greater extent than did previous monarchs".
Reasoning: Therefore, the statement directly addresses the question, highlighting George I's reliance on ministers compared to earlier monarchs.""",
        """Identify the question's focus: The question asks for the historian who linked Frédéric's friend's events in Poland to his maturity.
Find the relevant section: The passage mentions a friend, Woyciechowski, who returned to Poland to enlist in the uprising, and then discusses Frédéric's reaction to the events. The passage directly states that "Jachimecki ascribes to these events the composer's maturing..."
Reasoning: Therefore, the text clearly links Jachimecki to the idea that the events with the friend contributed to Frédéric's maturing.""",
        """Identify the question's focus: The question asks about who overran most of Mercia in the 900s.
Find the relevant section: The passage states that "Northumbria south of the Tyne, and most of Mercia, were overrun by the Vikings during the 9th century."
Reasoning: Therefore, the question specifies the 900s, which is the 9th century, and the passage provides the answer: Vikings.""",
        '''Identify the question's focus: The question asks for the meaning of the acronym "STAT" as used in the passage.
Find the relevant section: The passage mentions "subtherapeutic antibiotic treatment (STAT)" in the second sentence.
Reasoning: Therefore, the passage directly defines "STAT" as "subtherapeutic antibiotic treatment."''',
        """Identify the question's focus: The question asks about what physics has broadly agreed on the definition of.
Find the relevant section: The passage states that "there is no broad consensus as to a general definition of matter" in physics.
Reasoning: Therefore, physics has not broadly agreed on the definition of matter. The passage does not mention any other term or concept that physics has broadly agreed on the definition of.""",
        """Identify the question's focus: The question asks for the name of a "unit of play" in a database.
Find the relevant section: The passage describes database transactions as "a unit of work" encapsulating multiple operations.
Reasoning: Therefore, there is no mention of "play" or any game-related concepts in the passage, and the passage provides no information to answer the question about a "unit of play" in a database.""",
    ]

    instruction = """This task is about writing a correct answer for the reading comprehension task. Based on the information provided in a given passage, you should identify the shortest continuous text span from the passage that serves as an answer to the given question. Avoid answers that are incorrect or provide incomplete justification for the question. First, generate your explanations and thought process before generating the final answer. If you cannot find the answer from the passage for the given question, then generate the <no_answer> tag as the final answer."""

    icl_input = f"{instruction}"
    for idx, context_example in enumerate(contexts):
        icl_input = f"{icl_input}\n\nPassage_{idx+1}: {context_example}\nQuestion_{idx+1}: {questions[idx]}\nExplanations and Thought Process_{idx+1}: {explanations[idx]}\nFinal Answer: {gold_answers[idx]}"

    print(icl_input)

    test_passage = """The Panthers offense, which led the NFL in scoring (500 points), was loaded with talent, boasting six Pro Bowl selections. Pro Bowl quarterback Cam Newton had one of his best seasons, throwing for 3,837 yards and rushing for 636, while recording a career-high and league-leading 45 total touchdowns (35 passing, 10 rushing), a career-low 10 interceptions, and a career-best quarterback rating of 99.4. Newton's leading receivers were tight end Greg Olsen, who caught a career-high 77 passes for 1,104 yards and seven touchdowns, and wide receiver Ted Ginn, Jr., who caught 44 passes for 739 yards and 10 touchdowns; Ginn also rushed for 60 yards and returned 27 punts for 277 yards. Other key receivers included veteran Jerricho Cotchery (39 receptions for 485 yards), rookie Devin Funchess (31 receptions for 473 yards and five touchdowns), and second-year receiver Corey Brown (31 receptions for 447 yards). The Panthers backfield featured Pro Bowl running back Jonathan Stewart, who led the team with 989 rushing yards and six touchdowns in 13 games, along with Pro Bowl fullback Mike Tolbert, who rushed for 256 yards and caught 18 passes for another 154 yards. Carolina's offensive line also featured two Pro Bowl selections: center Ryan Kalil and guard Trai Turner."""
    test_question = "Who started at tight end for the Panthers?"
    test_input = (
        f"{icl_input}\n\nPassage_11: {test_passage}\nQuestion_11: {test_question}\nExplanations and Thought Process_11:"
    )
    response = model.generate_content(test_input, generation_config=generation_config)
    text = response.text
    print(text)

    dataset = load_dataset("rajpurkar/squad_v2", split="validation")
    for row in dataset:
        print(row)
        print(row.keys())
        break

    print(len(dataset))

    squad_results = []
    cut_off = 797
    for idx, row in enumerate(dataset):
        if idx > cut_off:
            context = row["context"]
            question = row["question"]
            squad_input = f"{icl_input}\n\nPassage_11: {context}\nQuestion_11: {question}\nExplanations and Thought Process_11:"
            try:
                response = model.generate_content(squad_input, generation_config=generation_config)
                squad_results.append(
                    {
                        "context": context,
                        "question": question,
                        "answers": row["answers"],
                        "gemini-1.0-pro-latest_answer_cot": response.text,
                        "id": row["id"],
                    }
                )
                print("processed", idx)
            except Exception as e:
                # ValueError: The `response.text` quick accessor only works when the response contains a valid `Part`, but none was returned. Check the `candidate.safety_ratings` to see if the response was blocked.
                try:
                    time.sleep(1)
                    response = model.generate_content(squad_input, generation_config=generation_config)
                    squad_results.append(
                        {
                            "context": context,
                            "question": question,
                            "answers": row["answers"],
                            "gemini-1.0-pro-latest_answer_cot": response.text,
                            "id": row["id"],
                        }
                    )
                    print("processed", idx)

                except Exception as e:
                    squad_results.append(
                        {
                            "context": context,
                            "question": question,
                            "answers": row["answers"],
                            "gemini-1.0-pro-latest_answer_cot": "<API_failed>",
                            "id": row["id"],
                        }
                    )
                    print(f"skipped this idx {idx}")

    with open(FLAGS.output_file, "w") as f:
        json.dump(squad_results, f)


if __name__ == "__main__":
    app.run(main)
