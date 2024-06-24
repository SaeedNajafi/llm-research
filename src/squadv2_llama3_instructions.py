from src.general_utils import white_space_fix

# Example context, question from squad.
contexts = [
    """Architecturally, the school has a Catholic character.\
        Atop the Main Building's gold dome is a golden statue of the Virgin Mary.\
        Immediately in front of the Main Building and facing it, is a copper statue\
        of Christ with arms upraised with the legend "Venite Ad Me Omnes".\
        Next to the Main Building is the Basilica of the Sacred Heart.\
        Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.\
        It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared\
        to Saint Bernadette Soubirous in 1858.\
        At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome),\
        is a simple, modern stone statue of Mary.""",
    """Estonian belongs to the Finnic branch of the Uralic languages, along with Finnish, Karelian,\
        and other nearby languages. The Uralic languages do not belong to the Indo-European languages.\
        Estonian is distantly related to Hungarian and to the Sami languages.""",
    """Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer,\
        songwriter, record producer and actress. Born and raised in Houston, Texas, she performed\
        in various singing and dancing competitions as a child, and rose to fame in the late 1990s\
        as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles,\
        the group became one of the world's best-selling girl groups of all time. Their hiatus saw\
        the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as\
        a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles\
        "Crazy in Love" and "Baby Boy".""",
    """The ISO 216 system used in most other countries is based on the surface area of a sheet of paper,\
    not on a sheet's width and length. It was first adopted in Germany in 1922 and generally spread as\
    nations adopted the metric system. The largest standard size paper is A0 (A zero), measuring one\
    square meter (approx. 1189 × 841 mm). Two sheets of A1, placed upright side by side fit exactly\
    into one sheet of A0 laid on its side. Similarly, two sheets of A2 fit into one sheet of A1 and\
    so forth. Common sizes used in the office and the home are A4 and A3 (A3 is the size of two A4 sheets).""",
    """During the rule of the succeeding Hanoverian dynasty, power was gradually exercised more by parliament\
    and the government. The first Hanoverian monarch, George I, relied on his ministers to a greater extent\
    than did previous monarchs. Later Hanoverian monarchs attempted to restore royal control over legislation:
    George III and George IV both openly opposed Catholic Emancipation and asserted that to grant assent to a\
    Catholic emancipation bill would violate the Coronation Oath, which required the sovereign to preserve\
    and protect the established Church of England from Papal domination and would grant rights to individuals\
    who were in league with a foreign power which did not recognise their legitimacy. However, George IV\
    reluctantly granted his assent upon the advice of his ministers. Thus, as the concept of ministerial\
    responsibility has evolved, the power to withhold royal assent has fallen into disuse, both\
    in the United Kingdom and in the other Commonwealth realms.""",
    '''Chopin's successes as a composer and performer opened the door to western Europe for him,\
    and on 2 November 1830, he set out, in the words of Zdzisław Jachimecki, "into the wide world,\
    with no very clearly defined aim, forever." With Woyciechowski, he headed for Austria, intending
    to go on to Italy. Later that month, in Warsaw, the November 1830 Uprising broke out, and Woyciechowski\
    returned to Poland to enlist. Chopin, now alone in Vienna, was nostalgic for his homeland, and wrote to\
    a friend, "I curse the moment of my departure." When in September 1831 he learned, while travelling\
    from Vienna to Paris, that the uprising had been crushed, he expressed his anguish in the pages of\
    his private journal: "Oh God! ... You are there, and yet you do not take vengeance!" Jachimecki\
    ascribes to these events the composer's maturing "into an inspired national bard who intuited the past,\
    present and future of his native Poland."''',
    """Each of these four dialects was associated with an independent kingdom on the island. Of these,\
    Northumbria south of the Tyne, and most of Mercia, were overrun by the Vikings during the 9th century.\
    The portion of Mercia that was successfully defended, and all of Kent, were then integrated into Wessex\
    under Alfred the Great. From that time on, the West Saxon dialect (then in the form now known as Early\
    West Saxon) became standardised as the language of government, and as the basis for the many works of\
    literature and religious materials produced or translated from Latin in that period.""",
    """Exposure to antibiotics early in life is associated with increased body mass in humans and mouse models.\
    Early life is a critical period for the establishment of the intestinal microbiota and for metabolic\
    development. Mice exposed to subtherapeutic antibiotic treatment (STAT)– with either penicillin,\
    vancomycin, penicillin and vancomycin, or chlortetracycline had altered composition of the gut\
    microbiota as well as its metabolic capabilities. Moreover, research have shown that mice given\
    low-dose penicillin (1 μg/g body weight) around birth and throughout the weaning process had an\
    increased body mass and fat mass, accelerated growth, and increased hepatic expression of genes\
    involved in adipogenesis, compared to controlled mice. In addition, penicillin in combination with\
    a high-fat diet increased fasting insulin levels in mice. However, it is unclear whether or not\
    antibiotics cause obesity in humans. Studies have found a correlation between early exposure of\
    antibiotics (<6 months) and increased body mass (at 10 and 20 months). Another study found that\
    the type of antibiotic exposure was also significant with the highest risk of being overweight\
    in those given macrolides compared to penicillin and cephalosporin. Therefore, there is correlation\
    between antibiotic exposure in early life and obesity in humans, but whether or not there is a causal\
    relationship remains unclear. Although there is a correlation between antibiotic use in early life\
    and obesity, the effect of antibiotics on obesity in humans needs to be weighed against the beneficial\
    effects of clinically indicated treatment with antibiotics in infancy.""",
    """The term "matter" is used throughout physics in a bewildering variety of contexts: for example,\
    one refers to "condensed matter physics", "elementary matter", "partonic" matter, "dark" matter,\
    "anti"-matter, "strange" matter, and "nuclear" matter. In discussions of matter and antimatter,\
    normal matter has been referred to by Alfvén as koinomatter (Gk. common matter). It is fair to say\
    that in physics, there is no broad consensus as to a general definition of matter, and the term\
    "matter" usually is used in conjunction with a specifying modifier.""",
    """Database transactions can be used to introduce some level of fault tolerance and data integrity\
    after recovery from a crash. A database transaction is a unit of work, typically encapsulating a\
    number of operations over a database (e.g., reading a database object, writing, acquiring lock, etc.),\
    an abstraction supported in database and also other systems. Each transaction has well defined boundaries\
    in terms of which program/code executions are included in that transaction (determined by the\
    transaction's programmer via special transaction commands).""",
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

explanations = [
    """Identify the question's focus: The question asks specifically about the individual to whom the\
        Virgin Mary appeared in Lourdes.
Find the relevant section: The passage mentions a replica of the grotto at Lourdes, France,\
    where the Virgin Mary is said to have appeared. It also mentions that this appearance was to\
    Saint Bernadette Soubirous in 1858.
Reasoning: Therefore, we know the Virgin Mary appeared at Lourdes to Saint Bernadette Soubirous.""",
    """Identify the question's focus: The question asks for a Uralic language branch that does NOT\
        contain Estonian.
Find the relevant section: The passage states that Estonian belongs to the Finnic branch of the Uralic\
    languages.
Reasoning: Therefore, any Uralic language branch that is NOT the Finnic branch would be a valid answer.\
    However, the passage does not mention any other Uralic language branches besides the Finnic branch.\
        We only know that Estonian is distantly related to Hungarian and the Sami languages, but we don't\
        know if these are part of other Uralic branches. Thus, we lack the information to determine which\
        Uralic language branches do not contain Estonian.""",
    """Identify the question's focus: The question asks about Beyonce's rise to fame.
Find the relevant section: The passage mentions her early career, saying she "performed in various singing\
and dancing competitions as a child". The passage then states "she rose to fame in the late 1990s as lead\
    singer of R&B girl-group Destiny's Child".
Reasoning: Therefore, the statement directly answers the question, stating that her rise to fame began\
    in the late 1990s.""",
    """Identify the question's focus: The question asks about the adoption of the ISO 1189 system in Germany.
Find the relevant section: The passage mentions the ISO 216 system, not ISO 1189. The passage states that\
    the ISO 216 system was first adopted in Germany in 1922.
Reasoning: Therefore, the question asks about a system not discussed in the passage, making it unanswerable.""",
    """Identify the question's focus: The question asks about which Hanoverian monarch relied most heavily\
        on their ministers.
Find the relevant section: The passage mentions that Hanoverian monarchs gradually shifted power to\
    Parliament and the government, and that George I relied on his ministers "to a greater extent\
        than did previous monarchs".
Reasoning: Therefore, the statement directly addresses the question, highlighting George I's reliance\
    on ministers compared to earlier monarchs.""",
    """Identify the question's focus: The question asks for the historian who linked Frédéric's\
        friend's events in Poland to his maturity.
Find the relevant section: The passage mentions a friend, Woyciechowski, who returned to Poland to\
    enlist in the uprising, and then discusses Frédéric's reaction to the events. The passage directly\
        states that "Jachimecki ascribes to these events the composer's maturing..."
Reasoning: Therefore, the text clearly links Jachimecki to the idea that the events with the friend\
    contributed to Frédéric's maturing.""",
    """Identify the question's focus: The question asks about who overran most of Mercia in the 900s.
Find the relevant section: The passage states that "Northumbria south of the Tyne, and most of Mercia,\
    were overrun by the Vikings during the 9th century."
Reasoning: Therefore, the question specifies the 900s, which is the 9th century, and the passage provides\
    the answer: Vikings.""",
    '''Identify the question's focus: The question asks for the meaning of the acronym "STAT" as used in\
        the passage.
Find the relevant section: The passage mentions "subtherapeutic antibiotic treatment (STAT)" in the second\
    sentence.
Reasoning: Therefore, the passage directly defines "STAT" as "subtherapeutic antibiotic treatment."''',
    """Identify the question's focus: The question asks about what physics has broadly agreed on the\
        definition of.
Find the relevant section: The passage states that "there is no broad consensus as to a general definition\
    of matter" in physics.
Reasoning: Therefore, physics has not broadly agreed on the definition of matter. The passage does not\
    mention any other term or concept that physics has broadly agreed on the definition of.""",
    """Identify the question's focus: The question asks for the name of a "unit of play" in a database.
Find the relevant section: The passage describes database transactions as "a unit of work" encapsulating\
    multiple operations.
Reasoning: Therefore, there is no mention of "play" or any game-related concepts in the passage, and the\
    passage provides no information to answer the question about a "unit of play" in a database.""",
]

explanation_instruction = """This task is about writing a correct answer for the reading comprehension task.\
    Based on the information provided in a given passage, you should identify the shortest continuous text span\
        from the passage that serves as an answer to the given question. Avoid answers that are incorrect\
            or have incomplete justification. Generate your explanations and thought process before\
                generating the final answer. If you cannot find the answer from the passage for the given\
                    question, then generate the <no_answer> tag as the final answer."""
normal_instruction = """This task is about writing a correct answer for the reading comprehension task.\
    Based on the information provided in a given passage, you should identify the shortest continuous text\
        span from the passage that serves as an answer to the given question. Avoid answers that are incorrect\
            or have incomplete justification. If you cannot find the answer from the passage for the given\
                question, then generate the <no_answer> tag as the final answer."""

normal_instruction = white_space_fix(normal_instruction)

normal_icl_input = normal_instruction
for idx, context_example in enumerate(contexts):
    normal_icl_input += f"\n\nPassage_{idx+1}: {white_space_fix(context_example)}"
    normal_icl_input += f"\nQuestion_{idx+1}: {white_space_fix(questions[idx])}"
    normal_icl_input += f"\nFinal Answer_{idx+1}: {white_space_fix(gold_answers[idx])}"

explanation_instruction = white_space_fix(explanation_instruction)
explanation_icl_input = explanation_instruction
for idx, context_example in enumerate(contexts):
    explanation_icl_input += f"\n\nPassage_{idx+1}: {white_space_fix(context_example)}"
    explanation_icl_input += f"\nQuestion_{idx+1}: {white_space_fix(questions[idx])}"
    explanation_icl_input += f"\nExplanations and Thought Process_{idx+1}: {white_space_fix(explanations[idx])}"
    explanation_icl_input += f"\nFinal Answer_{idx+1}: {white_space_fix(gold_answers[idx])}"
