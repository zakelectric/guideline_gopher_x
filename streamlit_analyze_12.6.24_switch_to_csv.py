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
import asyncio
import datetime

#################### CONFIGURATION ####################

# Set test true or false so that the Gopher uses the test AWS bucket or the regular bucket
test = False

if test == True:
    BUCKET_NAME = 'guideline-gopher-x-test'
if test == False:
    BUCKET_NAME = 'guideline-gopher-x'

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
            model_name="gpt-4o-mini",
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
                "- loan_type (e.g., DSCR, Conventional, FHA, bank statement, VA, ITIN, etc.)\n"
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
            for loan criteria matches. 

            EXAMINE ALL TABLES AND MATRICES:
            - Carefully analyze any tables, matrices, or grids in the content
            - Check product matrices for specific loan types, LTVs, and credit score requirements
            - Verify eligibility tables for property types and loan purposes
            - Pay special attention to footnotes and exceptions in tables
            - Cross-reference between different tables to ensure all requirements align
            
            CRITICAL LOAN TYPE MATCHING:
            - First, scan all tables and text for the loan program type (e.g., DSCR, Bank Statement, Conventional, FHA)
            - The loan type must EXACTLY match what's requested in the query. Check program names and tables carefully
            - If you find ANY indication that the guideline contains a different loan program (e.g., DSCR guidelines when Bank Statements are requested), return matches=false
            - Many guidelines contain multiple programs - ensure you're looking at the correct program section
            
            TABLE ANALYSIS REQUIREMENTS:
            1. Find the specific product matrix/table for the requested loan type
            2. Locate the exact row/column intersection for:
            - Credit score
            - LTV/CLTV
            - Property type
            - Occupancy
            - Loan purpose
            3. Check overlay matrices or adjustment tables for additional restrictions
            4. Verify all footnotes and exceptions apply to this scenario
            
            VERIFICATION CHECKLIST (All must be true for a match):
            1. Loan Program: Must be explicitly listed in tables/matrices for the requested loan type
            2. Documentation: Tables must specifically show this documentation type is allowed
            3. Property Type: Must appear as eligible in property type matrices
            4. Loan Purpose: Must be listed as permitted in program matrices
            5. LTV: Must be explicitly allowed in LTV tables/matrices for this scenario
            6. Credit Score: Must meet minimums shown in credit score tables/matrices
            7. Occupancy: Must be listed as eligible in occupancy tables
    

            Return a VALID JSON with:
            - name of investor: string
            - matches: boolean
            - confidence_score: 0-100
            - relevant_details: string
            - restrictions: array of restrictions
            - credit score: minimum credit score for loan product that matches criteria's loan to value, credit score, property type, purpose, and any additional criteria
            - loan to value: maximum ltv for loan product that matches criteria's loan to value, credit score, property type, purpose, and any additional criteria"""),
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

################################################################################################################################################################################
    async def load_and_query_investor(self, s3_client, bucket: str, investor_prefix: str, embeddings, query: str, structured_criteria: dict, llm, guidelines_analyzer_prompt) -> Dict:
        #st.write("DEBUG 1")
        #st.write("BUCKET:", bucket)
        # timenow = datetime.datetime.now()
        # utc_time = timenow.astimezone(datetime.timezone.utc)
        # formatted_utc_time = utc_time.strftime("%Y-%m-%d %H:%M:%S%z")
        #st.write(f"DEF LOAD AND QUERY INVESTOR, AT START: {timenow}")
        try:
            # Create temp dir for this investor's files
            with tempfile.TemporaryDirectory() as temp_dir:
                #st.write("INVESTOR PREFIX:", investor_prefix)
                # Download .faiss and .pkl files
                for ext in ['.faiss', '.pkl']:
                    #st.write("EXT:", ext)
                    file_key = f"{investor_prefix}{"index"}{ext}"
                    #st.write("FILE KEY:", file_key)
                    local_path = os.path.join(temp_dir, f"index{ext}")
                    #st.write("LOCAL PATH:", local_path)
                    #st.write("TEMP DIR:", temp_dir)
                    await asyncio.to_thread(s3_client.download_file,bucket, file_key, local_path)
                    #st.write("DEBUG 3")
                    # timenow = datetime.datetime.now()
                    # utc_time = timenow.astimezone(datetime.timezone.utc)
                    # formatted_utc_time = utc_time.strftime("%Y-%m-%d %H:%M:%S%z")
                    #st.write(f"DEF LOAD AND QUERY INVESTOR, DOWNLOAD VECTOR: {timenow}")

                    try:
                        #st.write("Attempting to load vector store...")
                        self.vector_store = await asyncio.to_thread(FAISS.load_local, temp_dir, self.embeddings, allow_dangerous_deserialization=True)
                        #st.write("Vector store loaded successfully:", self.vector_store)
                    except Exception as e:
                        st.write("")
                       # st.write(f"Error type: {type(e)}")
                       # st.write(f"Error message: {str(e)}")
                
                    # Try to read the first few bytes of each file
                    for file in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, file)
                        try:
                            with open(file_path, 'rb') as f:
                                first_bytes = f.read(100)
                              #  st.write(f"First bytes of {file}: {first_bytes[:20]}")
                        except Exception as e:
                            st.write(f"Error reading {file}: {str(e)}")
                
                # Search
                try:
                    relevant_chunks = await asyncio.to_thread(self.vector_store.similarity_search, query, k=10)
                    st.write("QUERY:", query)
                    st.write("\nRELEVANT CHUNKS:")
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.write(f"\nChunk {i}:")
                        st.write(chunk.page_content)
                        st.write("Metadata:", chunk.metadata)
                        st.write("---")
                except Exception as e:
                    st.write("Error with relevant chunks:", e)

                # Process results
                results = []
                for chunk in relevant_chunks:
                    try:
                        analysis_response = await asyncio.to_thread(llm.invoke,
                            guidelines_analyzer_prompt.format(
                                criteria=json.dumps(structured_criteria),
                                content=chunk.page_content
                            )
                        )
                        # print("LLM INVOKED SUCCESSFULLY")
                        # st.write("DEBUG - LLM Invoked Successfully")
                    except Exception as e:
                        print(f"LLM INVOCATION ERROR: {str(e)}")
                        st.write(f"DEBUG - LLM Invocation Error: {str(e)}")
                        continue
                    
                    # Add explicit debugging here
                    print("ANALYSIS RESPONSE:", analysis_response)  # Using print for immediate output
                    #st.write("DEBUG - Analysis Response:", analysis_response)  # Using st.write for Streamlit display
                    
                    try:
                        # Clean up the JSON from markdown content
                        raw_content = analysis_response.content
                        # print("RAW CONTENT:", raw_content)  # Debug raw content
                        # st.write("DEBUG - Raw Content:", raw_content)
                        
                        json_str = raw_content.split("```json")[1].split("```")[0].strip()
                        # print("JSON STRING:", json_str)  # Debug JSON string
                        # st.write("DEBUG - JSON String:", json_str)
                        
                        analysis = json.loads(json_str)
                        # print("PARSED ANALYSIS:", analysis)  # Debug parsed analysis
                        # st.write("DEBUG - Parsed Analysis:", analysis)
                        
                    except Exception as e:
                        print(f"ERROR PARSING: {str(e)}")  # Debug error
                        st.write(f"DEBUG - Error Parsing: {str(e)}")
                        continue
                
                    if analysis and analysis.get('matches', False):
                        results.append({
                            "name of investor": chunk.metadata.get("investor", "Unknown"),
                            "confidence": analysis.get('confidence_score', 0),
                            "details": analysis.get('relevant_details', ''),
                            "restrictions": analysis.get('restrictions', []),
                            "credit score": analysis.get('credit score', 0),
                            "loan to value": analysis.get('loan to value', 0),
                            "source_url": chunk.metadata.get("s3_url", "")
                        })

                # timenow = datetime.datetime.now()
                # utc_time = timenow.astimezone(datetime.timezone.utc)
                # formatted_utc_time = utc_time.strftime("%Y-%m-%d %H:%M:%S%z")
                # st.write(f"DEF LOAD AND QUERY INVESTOR, AFTER ANALYSIS: {timenow}")
                
                return results
        
        except Exception as e:
            st.write(f"Error processing {investor_prefix}: {str(e)}")
            return []

###########################################################################################################################################################
    async def query_guidelines(self, query: str) -> Dict:
        

        structured_criteria_response = self.llm.invoke(
            self.query_parser_prompt.format(query=query)
        )
        structured_criteria = self._parse_llm_response(structured_criteria_response)

        # timenow = datetime.datetime.now()
        # utc_time = timenow.astimezone(datetime.timezone.utc)
        # formatted_utc_time = utc_time.strftime("%Y-%m-%d %H:%M:%S%z")
        # st.write(f"DEF QUERY GUIDELINES, AFTER QUERY PARSER PROMPT: {timenow}")
        
        if not structured_criteria:
            return {"error": "Failed to parse query"}
        
        # List all vector stores in S3
        response = self.s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix='vector_stores/',
            Delimiter='/'
        )
       # st.write("VECTOR STORES:", response)
        # timenow = datetime.datetime.now()
        # utc_time = timenow.astimezone(datetime.timezone.utc)
        # formatted_utc_time = utc_time.strftime("%Y-%m-%d %H:%M:%S%z")
        # st.write(f"DEF QUERY GUIDELINES, LIST ALL VECTOR STORES: {timenow}")
        
        if 'CommonPrefixes' not in response:
            return {"error": "No guidelines found"}
        
        # Create tasks for each investors
        tasks = []
        for prefix in response['CommonPrefixes']:
            investor_prefix = prefix['Prefix']
            task = self.load_and_query_investor(
                self.s3_client,
                BUCKET_NAME,
                investor_prefix,
                self.embeddings,
                query,
                structured_criteria,
                self.llm,
                self.guidelines_analyzer_prompt
            )
            tasks.append(task)

            # timenow = datetime.datetime.now()
            # utc_time = timenow.astimezone(datetime.timezone.utc)
            # formatted_utc_time = utc_time.strftime("%Y-%m-%d %H:%M:%S%z")
            # st.write(f"DEF QUERY GUIDELINES, TASKS APPENDING: {timenow}")
        
        # Run all queries in parallel
        all_results = await asyncio.gather(*tasks)
        # timenow = datetime.datetime.now()
        # utc_time = timenow.astimezone(datetime.timezone.utc)
        # formatted_utc_time = utc_time.strftime("%Y-%m-%d %H:%M:%S%z")
        # st.write(f"DEF QUERY GUIDELINES, TASKS APPENDED: {timenow}")
       # st.write("ALL RESULTS:", all_results)
        # Flatten results and remove duplicates
        results = [item for sublist in all_results for item in sublist]
        seen_investors = set()
        unique_results = []
        for result in results:
            if result['name of investor'] not in seen_investors:
                seen_investors.add(result['name of investor'])
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
    if uploaded_file is not None:
        st.write("ALERT: Uploaded file")
        with st.spinner("Processing PDF..."):
            investor, s3_url, vector_store_path = MortgageGuidelinesAnalyzer.load_and_process_pdf(uploaded_file)
            st.success(f"Successfully processed {investor}'s guidelines")
            st.write(f"S3 URL: {s3_url}")
            st.write(f"Vector store saved at: {vector_store_path}")
    
    # Query input
    query = st.text_area("Enter your query:")
    if st.button("Search Guidelines"):
        if query:

            # timenow = datetime.datetime.now()
            # utc_time = timenow.astimezone(datetime.timezone.utc)
            # formatted_utc_time = utc_time.strftime("%Y-%m-%d %H:%M:%S%z")
            #st.write(f"PUSHED BUTTON: {timenow}")

            with st.spinner("Analyzing guidelines..."):
                results = asyncio.run(st.session_state.analyzer.query_guidelines(query))
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.subheader("Query Understanding")
                    st.json(results["query_understanding"])
                    
                    st.subheader(f"Matching Investors ({results['total_matches']})")
                    for investor in results["matching_investors"]:
                        with st.expander(f"{investor['name of investor']} (Confidence: {investor['confidence']}%)"):
                            st.write("Details:", investor["details"])
                            st.write("Min credit score:", investor["credit score"])
                            st.write("Max LTV:", investor["loan to value"])
                            st.write("Restrictions:")
                            for restriction in investor["restrictions"]:
                                st.write(f"- {restriction}")
                            if investor["source_url"]:
                                st.write(f"Source: {investor['source_url']}")

if __name__ == "__main__":
    main()