import re, os, requests, nltk
nltk.download('punkt')  
nltk.download('stopwords')
from dotenv import load_dotenv
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pinecone import Pinecone


load_dotenv()

hf_token = os.environ.get('HF_TOKEN')
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
genai.configure(api_key=os.environ.get("GENAI_API_KEY"))

def generate_embedding(text: str) -> list[float]:

	response = requests.post(
		embedding_url,
		headers={"Authorization": f"Bearer {hf_token}"},
		json={"inputs": text})

	if response.status_code != 200:
		return "sorry"
     
	return response.json()


def pdf_reader(text:str) -> str:

    # text = extract_text(pdf_file).lower()
    achieve = text.split('achievements')[0] 
    certi = achieve.split('certificates')[0]  
    awards = certi.split("awards")[0]
    keywords = " ".join(re.findall(r'[a-zA-Z]\w+',awards))

#     get keywords
    token_text = word_tokenize(keywords)
    stop_words = stopwords.words('english')
    clean_text = []
    for i in token_text:
        if i not in stop_words:
            clean_text.append(i)
    clean_text = " ".join(clean_text)
    
    pattern = re.compile(r'[^a-zA-Z0-9\s]')
    clean_text = re.sub(pattern, '', clean_text).replace("\n", " ")
    
    # Define a regular expression pattern to match numbers
    pattern2 = r'\d+'

    # Remove numbers from the text using regex substitution
    text_without_numbers = re.sub(pattern2, '', clean_text)
    
    return text_without_numbers

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


def get_jobs_new(input_query:str,namespace:str, k=6):
    
    pine = Pinecone(api_key=os.environ.get('PINECONE_KEY'))
    index = pine.Index(os.environ.get('PINECONE_INDEX'))

    # input_query = pdf_reader(input_query)
    input_embed = generate_embedding(input_query)
    if input_embed == "sorry":
        return "upload","again" 

    pinecone_resp = index.query(vector=input_embed, top_k=k, include_metadata=True, namespace=namespace)
    if not pinecone_resp['matches']:
        # print(pinecone_resp)
        return "no jobs","no jobs"

    context = []
    scores = []
    for i in range(len(pinecone_resp['matches'])):

        scores.append(pinecone_resp['matches'][i]["score"] )
        context.append(pinecone_resp['matches'][i]['metadata'])
    
    return scores,context

if __name__ == "__main__":
    # def job_searcher(text:str,domain:str) :

    r_text = """
    mern stack developer github com ooh shit mern stack developer experience designing building robust web applications procient creating dynamic responsive user interfaces using react adept developing scalable server side solutions express js node js highly skilled database design management mongodb experienced optimizing code performance security adept collaborating cross functional teams deliver innovative user centric digital solutions strong problem solving abilities commitment staying updated emerging web development technologies best practices projects room booking system currently used meri college project details present newdelhi summary created web application using nodejs mysql booking rooms college enables user book rooms given time slots avid clas bookings model prototype stage used college beta testing pixel painter frontend web app using javascript html css helps creation pixelated art ease enables user create pixel art easily using toolkits available application
    """
    # sc, jds = get_jobs_new(r_text,"internship",4)
    sc = pdf_reader(r_text)

    print("Scores : ", sc)
    # print("\n Jobs : ", jds)
    # analysiss = []
    # missing_keys = []
    # # for jd in jds:
    # response = get_gemini_repsonse(r_text,jds[0]['description']).split(':')
    # analysiss.append(response[2])
    # missing_keys.append(response[1])

    # print("\n analysis : ", analysiss)
    # print("\n missing_keys : ", missing_keys)
    