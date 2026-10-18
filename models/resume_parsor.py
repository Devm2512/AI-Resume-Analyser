import re
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from utils.text_preprocessing import preprocessing_text
from utils.pdf_parser import extract_text_from_pdf
from langchain_ollama import ChatOllama


# def extract_skills_from_resume(resume_text, skill_list):


#     found_skill = []

#     for skill in skill_list:

#         skill_lower = skill.lower()

#         if re.search(rf"\b{re.escape(skill_lower)}\b", resume_text):

#             found_skill.append(skill)

#     return found_skill

load_dotenv()


def resume_parsor(text):
    Schema = [
        ResponseSchema(name = "Primary Skills", description= "idetify the primary skills from the resume"),
        ResponseSchema(name = "Secondary Skills", description = "identify the secondary skills from the resume"),
        ResponseSchema(name = "Tertiary Skills", description = "Identify the tertiary skills from the resume"),
        ResponseSchema(name = "Tools", description= "identify whether the candidate have knowledge of any tools"),
        ResponseSchema(name = "Soft Skills", description= "identify the soft skills in the candidate"),
        ResponseSchema(name = "Domain", description= "Identify the domain of the candidate")
    ]

    ResumeParser = StructuredOutputParser.from_response_schemas(Schema)

    prompt = PromptTemplate(
        template = "Identify the skills from the given text\n {format_instructions} \n {Resume}",
        input_variables= ["Resume"],
        partial_variables= {"format_instructions": ResumeParser.get_format_instructions()}
    )

    # resume_prompt = prompt.invoke({"Resume": text})

    model = ChatOllama(
        model = "qwen2.5-coder:3b",
        temperature=0, format="json"
    )

    final_model = ChatOllama(model="mistral:latest", temperature=0, format="json")

    chain = prompt | final_model | ResumeParser

    result = chain.invoke({"Resume": text})

    return result

def data_of_resume():
    resume_path = "C:/Users/dewan/OneDrive/Desktop/AI Analyzer and Selector/Data/Data/Resume.pdf"
    raw_data = extract_text_from_pdf(resume_path)
    clean_text = preprocessing_text(raw_data)
    resume_parsed_data = resume_parsor(clean_text)


    return resume_parsed_data

if __name__ == "__main__":
    resume_test = data_of_resume()

    print("\n--- Parsed Resume Output ---\n")
    print(resume_test)