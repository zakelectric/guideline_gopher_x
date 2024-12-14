import streamlit as st
import pickle
import os
import base64
import time
import re
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import boto3
from botocore.exceptions import NoCredentialsError
import logging
import pandas as pd
import json
from typing import List, Dict
import tempfile


#################### CONFIGURATION ####################

# Set test true or false so that the Gopher uses the test AWS bucket or the regular bucket
test = True

if test == True:
    BUCKET_NAME = 'guidelinegopher-test'
if test == False:
    BUCKET_NAME = 'guidelinegopher'

st.set_page_config(layout="wide")


st.markdown(
    """
    <link rel="canonical" href="https://guidelinegopher.com/" />
    """,
    unsafe_allow_html=True
)

# OpenAI API key
api_key = st.secrets["OPENAI_API_KEY"]
#os.environ["OPENAI_API_KEY"] = api_key



################## '--' GUIDELINE GOPHER X '--' ##################
################## '--' GUIDELINE GOPHER X '--' ##################
################## '--' GUIDELINE GOPHER X '--' ##################



def upload_to_aws(pdf, BUCKET_NAME, store_name_pdf):
    s3 = boto3.client('s3', aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                      aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
    
    
    try:
        s3.upload_fileobj(pdf, BUCKET_NAME, store_name_pdf)
        return True
    except FileNotFoundError:
        return False
    except NoCredentialsError:
        return False
    

def generate_presigned_url(BUCKET_NAME, store_name_pdf, expiration=7200):
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': BUCKET_NAME, 'Key': store_name_pdf},
                                                    ExpiresIn=expiration)
    except Exception as e:
        st.error(f"Error generating URL: {e}")
        return None
    return response



class MortgageGuidelinesAnalyzer:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0,
            openai_api_key=openai_api_key
        )
        self.vector_store = None
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.getenv('AWS_BUCKET_NAME')
        
        self.query_parser_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a mortgage guidelines expert. Extract key loan criteria from queries "
                "into a structured format. Consider all possible ways these criteria might be expressed.\n\n"
                "Return a VALID JSON object with these fields:\n"
                "- loan_type (e.g., DSCR, Conventional, FHA, etc.)\n"
                "- purpose (Purchase, Refinance, Cash-out Refi, etc.)\n"
                "- ltv (numerical value or null)\n"
                "- credit_score (numerical value or null)\n"
                "- property_type (SFR, Multi-family, etc.)\n"
                "- additional_criteria (array of other important factors)"
            )),
            ("human", "{query}")
        ])
        
        self.guidelines_analyzer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mortgage guidelines expert analyzing provided guidelines 
            for loan criteria matches. Return a VALID JSON with:
            - matches: boolean
            - confidence_score: 0-100
            - relevant_details: string
            - restrictions: array of restrictions
            - credit_score: minimum score
            - loan_to_value: maximum ltv"""),
            ("human", "Query criteria: {criteria}\n\nGuideline content: {content}")
        ])

    def _parse_llm_response(self, response):
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    return None
            return None

    def upload_to_s3(self, file, filename: str) -> str:
        """Upload file to S3 and return the URL"""
        try:
            self.s3_client.upload_fileobj(file, self.bucket_name, f"guidelines/{filename}")
            return f"s3://{self.bucket_name}/guidelines/{filename}"
        except Exception as e:
            st.error(f"Failed to upload to S3: {str(e)}")
            return None

    def save_vector_store(self, investor_name: str):
        """Save vector store to local and push to GitHub"""
        if self.vector_store:
            local_path = f"vector_stores/{investor_name}"
            self.vector_store.save_local(local_path)
            # Add GitHub push logic here if needed
            return local_path
        return None

    def load_and_process_pdf(self, uploaded_file):
        """Process single uploaded PDF"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.flush()
            
            investor = os.path.splitext(uploaded_file.name)[0]
            
            loader = PyPDFLoader(tmp_file.name)
            pdf_docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            chunks = text_splitter.split_documents(pdf_docs)
            
            for chunk in chunks:
                chunk.metadata.update({
                    "investor": investor,
                    "source": uploaded_file.name
                })
            
            # Upload to S3
            s3_url = self.upload_to_s3(uploaded_file, uploaded_file.name)
            if s3_url:
                for chunk in chunks:
                    chunk.metadata["s3_url"] = s3_url
            
            # Create/update vector store
            if self.vector_store:
                self.vector_store.add_documents(chunks)
            else:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            
            # Save vector store
            vector_store_path = self.save_vector_store(investor)
            
            os.unlink(tmp_file.name)
            return investor, s3_url, vector_store_path

    def query_guidelines(self, query: str) -> Dict:
        if not self.vector_store:
            return {"error": "No guidelines loaded yet"}
            
        structured_criteria_response = self.llm.invoke(
            self.query_parser_prompt.format(query=query)
        )
        structured_criteria = self._parse_llm_response(structured_criteria_response)
        
        if not structured_criteria:
            return {"error": "Failed to parse query"}
        
        relevant_chunks = self.vector_store.similarity_search(query, k=10)
        
        results = []
        for chunk in relevant_chunks:
            st.write("Starting chunk processing")
            
            try:
                # Step 1: Format the messages
                messages = await self.guidelines_analyzer_prompt.format_messages(
                    criteria=json.dumps(structured_criteria),
                    content=chunk.page_content
                )
                st.write("Messages formatted successfully")
                
                # Step 2: Invoke the LLM
                analysis_response = await self.llm.invoke(messages)
                st.write("LLM invoked successfully")
                st.write("Response type:", type(analysis_response))
                st.write("Response content:", analysis_response.content)
                
                # Step 3: Parse the response
                analysis = await asyncio.to_thread(self._parse_llm_response, analysis_response.content)
                st.write("Analysis parsed successfully")
        
        seen_investors = set()
        unique_results = []
        for result in results:
            if result['investor'] not in seen_investors:
                seen_investors.add(result['investor'])
                unique_results.append(result)
        
        return {
            "query_understanding": structured_criteria,
            "matching_investors": unique_results,
            "total_matches": len(unique_results)
        }

def main():
    st.title("Mortgage Guidelines Analyzer")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        st.session_state.analyzer = MortgageGuidelinesAnalyzer(openai_api_key)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Guidelines PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            investor, s3_url, vector_store_path = st.session_state.analyzer.load_and_process_pdf(uploaded_file)
            st.success(f"Successfully processed {investor}'s guidelines")
            st.write(f"S3 URL: {s3_url}")
            st.write(f"Vector store saved at: {vector_store_path}")
    
    # Query input
    query = st.text_area("Enter your query:")
    if st.button("Search Guidelines"):
        if query:
            with st.spinner("Analyzing guidelines..."):
                results = st.session_state.analyzer.query_guidelines(query)
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.subheader("Query Understanding")
                    st.json(results["query_understanding"])
                    
                    st.subheader(f"Matching Investors ({results['total_matches']})")
                    for investor in results["matching_investors"]:
                        with st.expander(f"{investor['investor']} (Confidence: {investor['confidence']}%)"):
                            st.write("Details:", investor["details"])
                            st.write("Restrictions:")
                            for restriction in investor["restrictions"]:
                                st.write(f"- {restriction}")
                            if investor["source_url"]:
                                st.write(f"Source: {investor['source_url']}")

if __name__ == "__main__":
    main()