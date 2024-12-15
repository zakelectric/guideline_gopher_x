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
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_csv_agent
import json
import boto3
from botocore.exceptions import NoCredentialsError
import logging
import pandas as pd
import tempfile
import asyncio
import datetime
from io import StringIO
from typing import List, Dict


#################### CONFIGURATION ####################

test = False
BUCKET_NAME = 'guideline-gopher-x-test' if test else 'guideline-gopher-x'

st.set_page_config(layout="wide")

st.markdown(
    """
    <link rel="canonical" href="https://guidelinegopher.com/" />
    """,
    unsafe_allow_html=True
)

api_key = st.secrets["OPENAI_API_KEY"]

class MortgageGuidelinesAnalyzer:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key
    )

        self.s3_client = boto3.client('s3', 
                                    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                                    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
        self.bucket_name = BUCKET_NAME

        try:
            csv_obj = self.s3_client.get_object(Bucket=BUCKET_NAME, Key='vector_stores/Non-Delegated ITIN Activator Matrix/combined_tables.csv')
            csv_content = csv_obj['Body'].read().decode('utf-8')
            self.tables_data = pd.read_csv(io.StringIO(csv_content))
            st.session_state['tables_loaded'] = True
        except Exception as e:
            st.error(f"Error loading tables data from S3: {e}")
            self.tables_data = None
            st.session_state['tables_loaded'] = False
            
        # Query parser prompt stays the same
        self.query_parser_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mortgage guidelines expert. Extract key loan criteria from queries 
            into a structured format. Consider all possible ways these criteria might be expressed.

            Return a VALID JSON object with these fields:
            - loan_type (e.g., DSCR, Conventional, FHA, bank statement, VA, ITIN, etc.)
            - purpose (Purchase, Refinance, Cash-out Refi, etc.)
            - ltv (numerical value or null)
            - credit_score (numerical value or null)
            - property_type (SFR, Multi-family, etc.)
            - loan_amount (numerical value or null)
            - dscr_value (numerical value or null, only for DSCR loans)
            - additional_criteria (array of other important factors)"""),
            ("human", "{query}")
        ])

         # Prompt for analyzing tables and guidelines
        self.guidelines_analyzer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mortgage guidelines expert. Analyze the provided guideline 
            content and determine if it matches the loan criteria. Consider both explicit and 
            implicit requirements. Look for:
            
            1. Direct matches of numerical criteria (LTV, credit score, etc.)
            2. Program eligibility (loan type, purpose)
            3. Property type restrictions
            4. Any other relevant restrictions or requirements
            5. Ensure that the credit score / FICO requirements meet the needs of the query
            6. LTV / loan-to-value requirements with a numerical value that is higher than the query requirement
            
            Return a VALID JSON object with:
            - matches: boolean
            - confidence_score: 0-100
            - relevant_details: string explaining the match or mismatch
            - restrictions: array of important caveats or restrictions
            - credit score: minimum credit score for program
            - loan to value: maximum loan to value for program
            
            IMPORTANT: Ensure the response is a VALID JSON that can be parsed by json.loads()"""),
            ("human", """Query criteria: {criteria}
            
            Guideline content: {content}""")
        ])

    async def load_and_query_investor(self, s3_client, bucket: str, investor_prefix: str, query: str, structured_criteria: dict):
        #st.write(f"Processing investor {investor_prefix} with criteria:", structured_criteria)
        try:
            # Load the vector stores
            with tempfile.TemporaryDirectory() as temp_dir:
                for ext in ['.faiss', '.pkl']:
                    file_key = f"{investor_prefix}{'index'}{ext}"
                    local_path = os.path.join(temp_dir, f"index{ext}")
                    await asyncio.to_thread(s3_client.download_file, bucket, file_key, local_path)
                    
                self.vector_store = await asyncio.to_thread(
                    FAISS.load_local, 
                    temp_dir, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                # Get relevant text chunks
                relevant_chunks = await asyncio.to_thread(
                    self.vector_store.similarity_search,
                    query,
                    k=5
                )

                # Use LLM to analyze each chunk thoroughly
                for chunk in relevant_chunks:
                    analysis_response = await asyncio.to_thread(self.llm.invoke,
                        self.guidelines_analyzer_prompt.format(
                            criteria=json.dumps(structured_criteria),
                            content=chunk.page_content
                        )
                    )
                
                st.write("ANALYSIS RESPONSE:", analysis_response)
                analysis = self._parse_llm_response(analysis_response)
                
                if analysis and analysis.get('matches', False):
                    return [{
                        "name of investor": relevant_chunks[0].metadata.get("investor", "Unknown"),
                        "confidence": analysis.get('confidence_score', 0),
                        "details": analysis.get('relevant_details', ''),
                        "restrictions": analysis.get('restrictions', []),
                        "credit score": analysis.get('credit score', 0),
                        "loan to value": analysis.get('loan to value', 0),
                        "footnotes": analysis.get('footnotes', []),
                        "source_url": relevant_chunks[0].metadata.get("s3_url", "")
                    }]
                    
        except Exception as e:
            st.error(f"Error in final analysis: {e}")
            return []
        
    async def _aggregate_results(self, results: List[Dict]) -> List[Dict]:
        """Aggregate and deduplicate results by investor."""
        # Simple implementation - keep unique investors
        seen_investors = set()
        unique_results = []
        for result in results:
            if result['investor'] not in seen_investors:
                seen_investors.add(result['investor'])
                unique_results.append(result)
        return unique_results

    async def query_guidelines(self, query: str):

        # Get JSON object from query
        structured_criteria_response = self.llm.invoke(
            self.query_parser_prompt.format(query=query)
        )
        # Turn JSON string into true JSON
        structured_criteria = self._parse_llm_response(structured_criteria_response)
                
        if not structured_criteria:

            return {"error": "Failed to parse query"}

        try:
            # List vector stores in S3
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='vector_stores/',
                Delimiter='/'
            )
            if 'CommonPrefixes' not in response:
                return {"error": "No guidelines found"}

            # Debug - print first task creation
            st.write("Creating first task with criteria:", structured_criteria)
            
            # Create tasks for each guideline in S3
            tasks = []
            for prefix in response['CommonPrefixes']:
                investor_prefix = prefix['Prefix']
                task = self.load_and_query_investor(
                    self.s3_client,
                    self.bucket_name,
                    investor_prefix,
                    query,
                    structured_criteria  # This is where structured_criteria is used
                )
                tasks.append(task)

            all_results = await asyncio.gather(*tasks)
            
            # Process results
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
        except Exception as e:
            st.error(f"Error in query processing: {str(e)}")
            return {"error": f"Query processing failed: {str(e)}"}

    def _parse_llm_response(self, response):
        try:
            # If it's a full response object with content attribute
            content = response.content if hasattr(response, 'content') else response
            
            # Find the JSON content between ```json and ```
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
                
            # If no markdown blocks, try to find JSON directly
            json_match = re.search(r'\{.*?\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
                
            return None
            
        except Exception as e:
            st.write(f"Error parsing LLM response: {str(e)}")
            st.write("Raw content:", content)
            return None

def main():
    st.title("Mortgage Guidelines Analyzer")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = MortgageGuidelinesAnalyzer(api_key)
    
    # Query input
    query = st.text_area("Enter your query:")
    if st.button("Search Guidelines"):
        if query:
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
                            if investor.get("footnotes"):
                                st.write("Footnotes:")
                                for footnote in investor["footnotes"]:
                                    st.write(f"- {footnote}")
                            if investor["source_url"]:
                                st.write(f"Source: {investor['source_url']}")

if __name__ == "__main__":
    main()