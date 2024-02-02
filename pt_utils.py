from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory
from langchain.prompts import PromptTemplate

# this is specific to Llama-2.
# system_prompt = """You are an AI-powered flight information assistant, including flight numbers, departure and arrival times, as well as information about any delays or cancellations. You can input the data in a structured format or provide it as a narrative description."""
system_prompt = """You are an AI-powered healthcare assistant, an advanced Medical information assistant. Include information such as medical history, current medications, allergies, and recent encounters.ensure provides a comprehensive overview of the patient's health status."""
demographic_system_prompt = """You are an AI-powered healthcare assistant, an advanced Medical information assistant. Include information such as patient Sex , Age, Race, Ethnicity,  and Comorbid condition in zip code."""
encounter_system_prompt = """You are an AI-powered healthcare assistant, an advanced Medical information assistant. Include information such as patient recent encounters."""
vedic_prompt1 = """Develop a system that utilizes either the client's birthday or name to generate personalized predictions. Incorporate astrological or numerological principles to provide insights into various aspects of the client's life, such as career, relationships, and health. Ensure the system is designed to offer positive and constructive guidance, taking into consideration the ethical implications of providing predictive information. Additionally, explore the possibility of incorporating user preferences or customization options for a more personalized and engaging experience."""
vedic_prompt2 = """
Design a system for personalized predictions using Vedic astrology principles. The system should take into account the client's birth date, time, and name to generate accurate and detailed insights. Incorporate key features such as planetary positions, zodiac signs, and house placements in the analysis. Implement the use of dashas and bhuktis for predicting specific life events, and consider any yogas formed in the birth chart. Additionally, explore the potential for suggesting auspicious timings (muhurta) based on the client's details. Provide a user-friendly interface for clients to input their information and receive personalized astrological predictions.
"""
#You are a learned Vedic astrologer making predictions based on your clients' horoscopes. The system should take into account the client's birth date, time, and name to generate accurate and detailed insights.
vedic_prompt = """
 Incorporate key features such as Moon Sign, Sun Sign , Birth Nakshatra, Lagna Rashi(Sign) , Effects of Planets in horoscope and Astrology Remedies.Suggest auspicious timings (muhurta) based on the client's details.
"""
vedic_prompt3 = """
Example:
Promot: Generate a personalized horoscope for John, born on 1990-05-15 in New York at 08:30 AM. Provide insights on my career.
Personalized Horoscope for John:

🌟 Love: Your relationships will be harmonious and full of joy this month.

💼 Career: A new career opportunity is on the horizon. Stay open to new challenges.

🌿 Health: Prioritize rest and relaxation for overall well-being.

Insights on my career: Favorable opportunities await you. Embrace changes and take calculated risks for professional growth.


"""
t = """
You are a learned Vedic astrologer making predictions based on your clients' horoscopes. 
Synthesize the knowledge of GPT with the following extracted parts of a document and a question, then create a final answer.

Here is an example of how to answer every question: "Will I get a job promotion this year?":

Tree branching:

Branch 1: Analyze the 10th house in the birth chart for career and status.

Branch 2: Examine the 10th house lord for career prospects. Consider the lord's strength based on shadbala.

Branch 3: Inspect planets placed in the 10th house and their influence on career.

Branch 4: Study the D10 dasamsa chart for career orientation by looking at the 10th house, its lord and planets placed within.

Branch 5: Evaluate the current mahadasha and antardasha lords for career development during their periods.

Branch 6: Analyze transits over natal 10th house, 10th lord and relevant planets to see career triggers.

"""

# insurance inquiries and applications. Prioritize user privacy and data security at all times. When users provide personal information, ensure that it is treated with the utmost care and confidentiality. Only request and use the minimum necessary information to assist with insurance-related queries. Do not store or share any personal data. Always follow industry-standard security practices and regulations when handling sensitive information.


def get_system_prompt(x=None):
    return {
        'a': demographic_system_prompt,
        'b': encounter_system_prompt,
        'x': system_prompt
    }.get(x, vedic_prompt)


def get_prompt_template(system_prompt=vedic_prompt, promptTemplate_type=None, history=False):
    # lama2 template
    if promptTemplate_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(
                input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(
                input_variables=["context", "question"], template=prompt_template)
    # Mistral prompt template
    elif promptTemplate_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(
                input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                B_INST
                + system_prompt
                + """
            
            Context: {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(
                input_variables=["context", "question"], template=prompt_template)
    elif promptTemplate_type == "vedic":
        prompt_template = (
            system_prompt
            + """
               
                Based on the 10th house, its lord, planets within, D10 chart analysis, current periods and transits - provide an in-depth analysis on the possibility of gaining a promotion this year.
                {context}
                {history}
                Human: {human_input}
                Chatbot:
                """
        )
        prompt = PromptTemplate(
            input_variables=["history", "human_input", "context"], template=prompt_template, k=5
        )
    memory = ConversationBufferMemory(
        input_key="question", memory_key="history")
    return (
        prompt,
        memory,
    )
