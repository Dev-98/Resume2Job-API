from dotenv import load_dotenv
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re, os, requests
from pinecone import Pinecone


load_dotenv()

hf_token = os.environ.get('HF_TOKEN')
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

def generate_embedding(text: str) -> list[float]:

	response = requests.post(
		embedding_url,
		headers={"Authorization": f"Bearer {hf_token}"},
		json={"inputs": text})

	if response.status_code != 200:
		raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
	return response.json()


def pdf_reader(text:str) -> str:

    # text = extract_text(pdf_file).lower()
    skill = text.split("skills")[1]

#     get keywords
    keywords = " ".join(re.findall(r'[a-zA-Z]\w+',skill.lower()))

    token_text = word_tokenize(keywords)
    stop_words = stopwords.words('english')
    clean_text = []
    for i in token_text:
        if i not in stop_words:
            clean_text.append(i)
    clean_text = " ".join(clean_text)
    
    pattern = re.compile(r'[^a-zA-Z0-9\s]')
    clean_text = re.sub(pattern, '', clean_text).replace("\n", "")
    
    # Define a regular expression pattern to match numbers
    pattern2 = r'\d+'

    # Remove numbers from the text using regex substitution
    text_without_numbers = re.sub(pattern2, '', clean_text)
    
    return clean_text

def get_gemini_repsonse(r_text:str,jd:str) :
    input = f"""Act like a skilled or very experience ATS(Application Tracking System)
          with a deep understanding of various software fields. Your task is to evaluate the resume based on the given job description.
          And provide an analysis of the resume based on the given job description and
          best assistance for improving the resumes. Point out the missing keywords in resume with high accuracy
          resume:{r_text}
          description:{jd}

          I want the response in dictionary format having the structure
          (MissingKeywords:[], Analysis :'')
        """
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content(input)
    return response.text


def get_jobs_new(input_query:str,namespace:str, k=4):
    
    pine = Pinecone(api_key=os.getenv('PINECONE_KEY'))
    index = pine.Index(os.getenv('PINECONE_INDEX'))

    # input_query = pdf_reader(input_query)
    input_embed = generate_embedding(input_query)

    pinecone_resp = index.query(vector=input_embed, top_k=k, include_metadata=True, namespace=namespace)
    if not pinecone_resp['matches']:
        # print(pinecone_resp)
        return "No Jobs Found, Maybe you should learn more skills and do more projects to get more jobs"

    context = []
    scores = []
    for i in range(len(pinecone_resp['matches'])):

        scores.append(pinecone_resp['matches'][i]["score"] )
        context.append(pinecone_resp['matches'][i]['metadata'])
    
    return scores,context

if __name__ == "__main__":
    # def job_searcher(text:str,domain:str) :

    r_text = """
    skills python javascript computer vision generative ai git github mlops kubernetes mongodb firebase db pinecone mysql machine learning deep learning google cloud platform microsoft azure vector db docker flask fastapi professional experience dataknobs ml engineer source contributor mlflow contributor august august contributed key functionality got merged administrator mlflow google cloud google cloud facilitator may july acquired proficiency docker mlops kubernetes kubernetes relevant projects sign language tutor march present used learning sign language fun interactive way chakla controller asphalt january january innovative racing game controlled unique physical interface round flat board blue square uses opencv computer vision techniques translate board movements game actions medsarthi january january helping seniors understand medications simple image upload voice enabled explanations education maharishi dayanand university rohtak bachelor computer science artificial intelligence rajokari institute technology dseu diploma information technology enabled service management
    """
    sc, jds = get_jobs_new(r_text,"internship",4)

    print("Scores : ", sc)
    print("\n Jobs : ", jds)