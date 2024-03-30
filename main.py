from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from function import *

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Hallo, ich bin Dev, wie geht's"


@app.route('/resume',methods=['GET','POST'])
def resume():

    try :

        if request.method == 'POST':
            # files = request.files['resume']
            text = request.form.get('resume_text').lower()
            domain = request.form.get('domain')
            # job_no = request.form.get('job_number',4)
            
            clean_text = pdf_reader(text)
            scores,jobs = get_jobs_new(clean_text,domain)

            analysiss = []
            missing_keys = []
            for jd in jobs:
                response = get_gemini_repsonse(clean_text,jd['description']).split(':')
                analysiss.append(response[2])
                missing_keys.append(response[1])

            output = {'missing' :  missing_keys,
                'analytics': analysiss,
                'score': scores,
                'jobs': jobs,
                }
            
            return jsonify(output), 200
    
    except Exception as e:
        return jsonify({"error":str(e)}), 500
    
if __name__ == '__main__':
	app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080))) 
