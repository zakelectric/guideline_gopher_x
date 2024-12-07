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
import tempfile
import asyncio
import datetime

#################### CONFIGURATION ####################

# Set test true or false so that the Gopher uses the test AWS bucket or the regular bucket
test = False

if test:
    BUCKET_NAME = 'guideline-gopher-x-test'
else:
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

class MortgageGuidelinesAnalyzer:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            openai_api_key=openai_api_key
        )
        self.vector_store = None
        self.s3_client = boto3.client('s3', 
                                    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                                    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
        self.bucket_name = BUCKET_NAME
        
        # Load CSV data from S3
        try:
            csv_obj = self.s3_client.get_object(Bucket=BUCKET_NAME, Key='vector_stores/COIN - Cashflow Only Investor Loan Product Matrix/combined_tables.csv')
            csv_content = csv_obj['Body'].read().decode('utf-8')
            self.tables_data = pd.read_csv(io.StringIO(csv_content))
            st.session_state['tables_loaded'] = True
        except Exception as e:
            st.error(f"Error loading tables data from S3: {e}")
            self.tables_data = None
            st.session_state['tables_loaded'] = False

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

        self.table_analyzer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mortgage guidelines expert analyzing loan criteria tables.
            
            EXAMINE TABLE DATA CAREFULLY:
            1. First verify the loan type matches exactly (e.g., DSCR vs Conventional)
            2. Find the intersection of:
                - Credit score
                - LTV/CLTV
                - Property type
                - Loan amount
                - Loan purpose
            3. Check all footnotes
            4. For DSCR loans, verify the DSCR threshold
            
            CRITICAL TABLE ANALYSIS RULES:
            - All numeric comparisons must be exact
            - Property types must match exactly
            - Loan amounts must be within specified ranges
            - LTV/CLTV must not exceed maximums
            - Credit scores must meet or exceed minimums
            
            Return a VALID JSON with:
            - matches: boolean (true if ALL criteria are met)
            - confidence_score: 0-100
            - max_ltv: maximum allowed LTV
            - min_credit_score: minimum required credit score
            - loan_amount_limits: {min: X, max: Y}
            - restrictions: array of restrictions
            - footnotes: array of relevant footnotes"""),
            ("human", "Criteria: {criteria}\n\nTable data: {table_data}")
        ])

        self.guidelines_analyzer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mortgage guidelines expert providing final analysis by combining
            table results with supporting text guidelines.

            CRITICAL ANALYSIS REQUIREMENTS:
            1. Table results are PRIMARY and deterministic
                - If tables show criteria aren't met, result MUST be no match
                - Numeric criteria from tables cannot be overridden
                - LTV and credit score limits from tables are final
            
            2. Text guidelines are SECONDARY and provide:
                - Additional restrictions
                - Clarifying details
                - Policy requirements
                - Documentation needs
                - Exceptions or special cases
            
            Return a VALID JSON with:
            - name of investor: string
            - matches: boolean
            - confidence_score: 0-100
            - relevant_details: string
            - restrictions: array of restrictions
            - credit score: minimum credit score that matches all criteria
            - loan to value: maximum ltv that matches all criteria
            - footnotes: array of relevant footnotes"""),
            ("human", "Table analysis: {table_results}\n\nText guidelines: {text_content}")
        ])

    def _extract_table_subset(self, criteria: dict) -> pd.DataFrame:
        """Extract relevant table rows based on criteria"""
        
        # Start with full dataset
        subset = self.tables_data.copy()
        
        # Apply filters based on available criteria
        if criteria.get('loan_type'):
            # Look for loan type in any column
            loan_type_mask = subset.apply(lambda x: x.astype(str).str.contains(criteria['loan_type'], case=False)).any(axis=1)
            subset = subset[loan_type_mask]
            
        if criteria.get('property_type'):
            # Look for property type in any column
            prop_type_mask = subset.apply(lambda x: x.astype(str).str.contains(criteria['property_type'], case=False)).any(axis=1)
            subset = subset[prop_type_mask]
        
        return subset

    async def analyze_tables(self, criteria: dict):
        """Analyze the CSV tables for matching criteria"""
        if self.tables_data is None:
            return {"error": "Tables data not available"}

        # Extract relevant subset of tables
        relevant_tables = self._extract_table_subset(criteria)
        
        # Convert tables data to string representation for the LLM
        table_str = relevant_tables.to_string()
        
        try:
            analysis = self.llm.invoke(
                self.table_analyzer_prompt.format(
                    criteria=json.dumps(criteria),
                    table_data=table_str
                )
            )
            return self._parse_llm_response(analysis)
        except Exception as e:
            st.error(f"Error analyzing tables: {e}")
            return None

    async def load_and_query_investor(self, s3_client, bucket: str, investor_prefix: str, query: str, structured_criteria: dict):
        try:
            # Create temp dir for vectors
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download vector store files
                for ext in ['.faiss', '.pkl']:
                    file_key = f"{investor_prefix}{'index'}{ext}"
                    local_path = os.path.join(temp_dir, f"index{ext}")
                    await asyncio.to_thread(s3_client.download_file, bucket, file_key, local_path)

                # Load vector store
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

                # Combine text chunks
                text_content = "\n".join([chunk.page_content for chunk in relevant_chunks])

                # First analyze tables
                table_results = await self.analyze_tables(structured_criteria)
                
                if not table_results or table_results.get("error"):
                    return []

                # Then combine with text analysis
                try:
                    final_analysis = await asyncio.to_thread(
                        self.llm.invoke,
                        self.guidelines_analyzer_prompt.format(
                            table_results=json.dumps(table_results),
                            text_content=text_content
                        )
                    )
                    
                    analysis = self._parse_llm_response(final_analysis)
                    
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

                return []

        except Exception as e:
            st.error(f"Error processing {investor_prefix}: {str(e)}")
            return []

    async def query_guidelines(self, query: str):
        # Parse query into structured criteria
        structured_criteria_response = self.llm.invoke(
            self.query_parser_prompt.format(query=query)
        )
        structured_criteria = self._parse_llm_response(structured_criteria_response)

        if not structured_criteria:
            return {"error": "Failed to parse query"}

        # List vector stores in S3
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix='vector_stores/',
            Delimiter='/'
        )

        if 'CommonPrefixes' not in response:
            return {"error": "No guidelines found"}

        # Create tasks for parallel processing
        tasks = []
        for prefix in response['CommonPrefixes']:
            investor_prefix = prefix['Prefix']
            task = self.load_and_query_investor(
                self.s3_client,
                self.bucket_name,
                investor_prefix,
                query,
                structured_criteria
            )
            tasks.append(task)

        # Run queries in parallel
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

def main():
    st.title("Mortgage Guidelines Analyzer")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = MortgageGuidelinesAnalyzer(api_key)
    
    # Show table loading status
    if 'tables_loaded' in st.session_state:
        if st.session_state['tables_loaded']:
            st.success("Tables data loaded successfully")
        else:
            st.error("Error loading tables data")
    
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