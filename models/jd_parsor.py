import json
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_ollama import ChatOllama


load_dotenv()


def parse_jd_with_llm(job_description):
    schema = [
        ResponseSchema(name = "must_have",description = "Identify and mention the must have skills according to a job"),
        ResponseSchema(name = "intermediate",description = "Identify and mention the intermediate skills according to a job"),
        ResponseSchema(name = "good_to_have",description = "Identify and mention the good to have skills according to a job")
    ]

    Parser = StructuredOutputParser.from_response_schemas(schema)

    prompt = PromptTemplate(
        template = "Identify the must have, intermedite, and good to have skills from the below text \n {format_instructions} \n {JD}",
        input_variables= ["JD"],
        partial_variables= {"format_instructions":Parser.get_format_instructions()}
    )

    final_prompt = prompt.invoke(job_description)

    # print(final_prompt)


    model = ChatOllama(
            model = "qwen2.5-coder:3b",
            temperature=0, format="json"
        )

    final_model = ChatOllama(model="mistral:latest", temperature=0, format="json")

    response = final_model.invoke(final_prompt)

    parsed_response = Parser.parse(response.content)

    # print(response.content)

    # chain = prompt | final_model | Parser

    # response = chain.invoke({"JD": job_description})

    # print("---------------------********************----------------------------------")
    # print(response)
    # print("---------------------********************----------------------------------")
    return parsed_response


if __name__ == "__main__":
    job_description = input("Enter Job Description: ")
    result = parse_jd_with_llm(job_description)

    print("\n--- PARSED JD OUTPUT ---\n")
    print(result)
