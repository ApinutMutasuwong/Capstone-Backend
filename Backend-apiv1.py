from flask import Flask, request, jsonify
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import os
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

os.environ["OPENAI_API_KEY"] = "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" 
openai_client = OpenAI()

client = MongoClient('mongodb+srv://Kung:Jan@capstone.u0tip.mongodb.net/?retryWrites=true&w=majority&appName=Capstone')
DB_NAME = "evals"
COLLECTION_NAME = "chunking"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,  
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),  
)

def rerank_with_cross_encoder(question, documents):
    pairs = [(question, doc.text if hasattr(doc, "text") else str(doc)) for doc in documents]
    scores = model.predict(pairs)
    reranked_docs = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
    return reranked_docs

def prompt_gpt(question, context):
    contacts="""1. Registration and Educational Services
                Responsibilities: Student registration, academic records, transcripts, graduation services.
                Website: www.reg.kmitl.ac.th
                Contacts:
                General Inquiries: Tel. 02-329-8203
                Admissions and Registration History: Tel. 02-329-8203
                Education Registration: Tel. 02-329-8203
                Academic Record Processing: Tel. 02-329-8202
                Academic Record Inspection and Certification: Tel. 02-329-8201
                Graduate Work: Tel. 02-329-8206

                2. Office of International Affairs (OIA)
                Responsibilities: Foreign student admissions, exchange programs, international partnerships.
                Website: www.oia.kmitl.ac.th
                Contacts:
                General Inquiry: Tel. (+66) 2-329-8000
                Foreign Relations Officer (Thitiya Pattanakit - MAI):
                Email: thitiya.pa@kmitl.ac.th
                Tel: (+66) 2-329-8000 ext. 2245

                3. Academic Administration and Quality Assurance (OAQ)
                Responsibilities: Academic policies, quality assurance, curriculum management.
                Email: academic@kmitl.ac.th
                Phone: 02-329-8000, 02-329-8136

                4. Undergraduate Admissions Office
                Responsibilities: Handling applications for undergraduate programs.
                Email: admission-reg@kmitl.ac.th
                Phone: 02-329-8000 ext. 3203-5
                Facebook: Admission.KMITL (https://www.facebook.com/Admission.KMITL/)

                5. Digital Information Management Office
                Responsibilities: IT support, student portals, system maintenance.
                Email: helpcenter@kmitl.ac.th
                Mobile: 091-190-6000
                Line ID: @helpcenter.kmitl

                6. Finance Office
                Responsibilities: Tuition fees, payment receipts, financial support.
                Email: kasaya.ba@kmitl.ac.th
                Phone: 02-329-8000 ext. 3220, 02-329-8120
                Mobile: 082-592-4542 (Kasaya)
                Website: https://finance.kmitl.ac.th/
                Office Location: 3rd Floor, Administration Building

                7. KMITL Hospital (Health Check-ups)
                Responsibilities: Student and faculty health check-ups.
                Website: https://kmch.kmitl.ac.th/
                Phone: 02-329-8143, 02-329-8000 ext. 3633

                8. Student Affairs Office (OSDA)
                Responsibilities: Student support, scholarships (e.g., กยศ.), alumni relations.
                Website: https://office.kmitl.ac.th/osda/
                Phone: 02-329-8000 ext. 3185

                9. Student Dormitory Office
                Responsibilities: Managing student housing and dormitory inquiries.
                Facebook: https://www.facebook.com/Dormitory.Of.KMITL
                Phone: 02-329-8145, 02-329-8164 (Open Mon-Sun, 08:30-16:30)

                10. Other KMITL Campuses & Institutes
                Chumphon Campus: www.pcc.kmitl.ac.th
                College of Materials Innovation & Technology: www.cmit.kmitl.ac.th
                College of Advanced Manufacturing Innovation: www.ami.kmitl.ac.th
                College of Industrial Management & Innovation: www.ciim-kmitl.online
                International Aviation Industry College: iaai.kmitl.ac.th
                Institute of Music Engineering: imse.kmitl.ac.th

                11. Faculty & Department Websites
                Engineering: www.eng.kmitl.ac.th
                Architecture & Design: www.aad.kmitl.ac.th
                Science: www.science.kmitl.ac.th
                Industrial Education & Technology: siet.kmitl.ac.th
                Agricultural Technology: www.agri.kmitl.ac.th
                IT & Digital Innovation: www.it.kmitl.ac.th
                Food Industry: foodindustry.kmitl.ac.th
                Business Administration: www.kbs.kmitl.ac.th
                Liberal Arts: la.kmitl.ac.th
                Medicine: md.kmitl.ac.th
                Dentistry: dent.kmitl.ac.th"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""
                You are a chatbot for KMITL, an intelligent assistant designed to provide accurate and helpful information about King Mongkut’s Institute of Technology Ladkrabang (KMITL). Your role is to assist students, faculty, and visitors by answering questions in a concise and friendly manner. 

                Here are the information about :

                {context}

                If a user asks for information that requires official confirmation, provide a general response and direct them to the university's official website or provide contact of the responsible office.

                If a technical question is outside your knowledge or the context doesn’t cover the answer, acknowledge it and suggest they look up KMITL's official website: http://www.kmitl.ac.th and provide a relevant responsible office. 

                List of KMITL Offices, Contacts and Responsibilities:

                {contacts}

                """
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response

app = Flask(__name__)

@app.route('/ask', methods=['GET'])
def ask():
    question = request.args.get('question', '')  
    canidate=vector_store.similarity_search(question, k=5)
    reranked_canidate=rerank_with_cross_encoder(question, canidate)
    
    response = prompt_gpt(question, reranked_canidate[:3])
    return jsonify({'message': response.choices[0].message.content})
    

if __name__ == '__main__':
    app.run(debug=True)