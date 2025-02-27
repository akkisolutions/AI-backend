import json
import os
from dotenv import load_dotenv # type: ignore
from groq import Groq
from utils.response import AppResponse, ServerErrorResponse, SuccessResponse

load_dotenv()

class GroqApi():

    def __init__(self):
        self.client = Groq()

    def call_llm(self) -> AppResponse:
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            print("completion", completion)

            return SuccessResponse("Successfully made the llm call", completion)
        except Exception as e:
            print("Failed to make the llm call", e)
            return ServerErrorResponse("Failed to make the llm call", e)
        
    def identify_names_from_prompt(self, known_names, query) -> AppResponse:
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "Simulate Persona:\nYou are an advanced natural language processing model specializing in named entity recognition (NER) with fuzzy matching. Your task is to extract names of people or pets from a given text. If a name is similar to one from a predefined list named known_names, return the matched name from the list instead. Additionally, replace any self-references (e.g., \"I\", \"me\", \"myself\", \"we\", \"us\") with \"CURRENT_USER\".\n\nTask:\nIdentify and extract the names of people or pets mentioned in the input text. Compare them against a predefined list of known names, allowing for minor spelling variations. If a match is found, return the standardized version from the list. Return only the names in a list format.\n\nSteps to Complete Task:\n\n    Analyze the sentence and extract proper nouns referring to people or pets.\n    Compare extracted names against the provided list of known names using fuzzy matching.\n    If a name is similar to a known name, return the standardized version from the list.\n    If the user refers to themselves (e.g., \"I\", \"me\", \"myself\", \"we\", \"us\"), replace it with \"CURRENT_USER\".\n    Exclude non-name words, titles, or descriptors (e.g., \"Mr.\", \"Dr.\", \"the cat\").\n    Return the extracted names in a structured list format.\n    If no names are found, return an empty list [].\n\nContext/Constraints:\n\n    The input text may contain multiple names. Ensure all are included.\n    A predefined list of known names will be provided. Prioritize matching against this list.\n    Use fuzzy matching to handle minor spelling variations (e.g., \"John\" should match \"Jon\").\n    Any self-reference by the user should always be replaced with \"CURRENT_USER\".\n    Do not include non-name words, locations, organizations, or objects.\n\nGoal:\nExtract and return only the names of people or pets as a clean, structured list. Names should be matched against the known list, and self-references should be replaced with \"CURRENT_USER\".\n\nFormat Output:\nA JSON dict with key user_names containing only the names. Example format:\n\n{ \"user_names\": [\"CURRENT_USER\", \"Emma\", \"Max\", \"Sarah\"]}"
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"known_names": known_names})
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                response_format={"type": "json_object"},
                stop=None,
            )

            data = completion.choices[0].message.content

            return SuccessResponse("Successfully made the llm call", json.loads(data))
        except Exception as e:
            print("Failed to make the llm call", e)
            return ServerErrorResponse("Failed to make the llm call", e)
