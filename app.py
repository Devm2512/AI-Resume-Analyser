from utils.pdf_parser import extract_text_from_pdf
from utils.text_preprocessing import preprocessing_text
from models.jd_parsor import parse_jd_with_llm
from models.match_engine import calculate_match_score
from models.resume_parsor import resume_parsor

resume_path = "C:/Users/dewan/OneDrive/Desktop/AI Analyzer and Selector/Data/Resume.pdf"


job_description = input("Enter Job Descrition:  ")


raw_data = extract_text_from_pdf(resume_path)

clean_text = preprocessing_text(raw_data)

resume_parsed_data = resume_parsor(clean_text)


job_description = job_description.lower()
print("------------------------------*********************************************************-----------------------------------------")
print(job_description)
print("------------------------------*********************************************************-----------------------------------------")

result = parse_jd_with_llm(job_description)

match_result = calculate_match_score(resume_parsed_data, result)

print("\n--- PARSED JD OUTPUT ---\n")
print(result)

print("\n--- CLEANED TEXT ---\n")
print(clean_text[:500])

if isinstance(result, dict) and "error" not in result:
    print("\n--- Structured Output ---")
    print("Must Have:", result.get("must_have"))
    print("Intermediate:", result.get("intermediate"))
    print("Good to Have:", result.get("good_to_have"))
else:
    print("\n Error in parsing:")
    print(result)

print("\n--- MATCH RESULT ---\n")
print(match_result)