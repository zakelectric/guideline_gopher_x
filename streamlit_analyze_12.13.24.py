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
        model_name="gpt-4o",
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


    async def analyze_tables(self, structured_criteria: dict):
        if self.tables_data is None:
            return {"error": "Tables data not available"}

        try:
            # Extract relevant subset
            if 'relevant_tables' not in st.session_state:
                st.session_state.relevant_tables = self.tables_data
                st.write("Tables extracted:")
                st.dataframe(st.session_state.relevant_tables)

            # Create agent with verbose output and callbacks
            if 'agent' not in st.session_state:
                class ThoughtTracer:
                    def on_agent_action(self, action, color="blue"):
                        st.markdown(f"🤔 **Thinking:** {action.log}", unsafe_allow_html=True)
                        if action.tool_input:
                            st.markdown(f"📝 **Planning to do:** {action.tool_input}")
                    
                    def on_agent_finish(self, output, color="green"):
                        st.markdown(f"✅ **Conclusion:** {output}", unsafe_allow_html=True)

                callbacks = [ThoughtTracer()]

                csv_buffer = StringIO()
                st.session_state.relevant_tables.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.session_state.agent = create_csv_agent(
                    llm=self.llm,
                    path=csv_buffer,
                    verbose=True,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                    prefix="""You are analyzing mortgage guidelines data.
                    Read the data and answer questions directly about specific loan programs.
                    Start at the bottom row of the relevant section and iterate UP to find minimum credit scores and maximum ltv.
                    Do not explain your process or write code.
                    Return ONLY the values found, formatted as JSON."""
                )
                st.write("Agent created with comprehensive analysis enabled")

            with st.expander("View Analysis Process", expanded=True):
                st.write("🔍 Starting detailed analysis...")
                
                # The analysis query is separate from the agent creation
                analysis_query = f"""For loan type: {structured_criteria['loan_type']}

                Find a minimum credit score that is less than {structured_criteria['credit_score']}
                Find a maximum ltv that is higher than {structured_criteria['ltv']}
                Find and return only this JSON:
                {{
                    "min_credit_score": [minimum FICO found],
                    "max_ltv": [highest LTV found],
                    "loan_amounts": {{
                        "min": [minimum amount],
                        "max": [maximum amount]
                    }}
                }}"""
                
                result = st.session_state.agent.run(analysis_query)
                st.write("📊 Analysis complete!")

            if isinstance(result, str):
                try:
                    json_match = re.search(r'\{.*\}', result, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))
                except Exception as e:
                    st.write(f"Error parsing agent response: {e}")
                    return {
                        "matches": False,
                        "confidence_score": 0,
                        "error": "Failed to parse agent response"
                    }

            return result
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}


    def _extract_table_subset(self, criteria: dict) -> pd.DataFrame:
        """Extract relevant table rows based on criteria"""
        
        subset = self.tables_data.copy()
        st.write("Initial table size:", len(subset))
        
        # Apply filters based on available criteria
        if criteria.get('loan_type'):
            # Look for loan type in any column
            loan_type_mask = subset.apply(lambda x: x.astype(str).str.contains(criteria['loan_type'], case=False)).any(axis=1)
            subset = subset[loan_type_mask]
            st.write(f"After loan type filter size: {len(subset)}")
        
        if criteria.get('property_type'):
            # Look for property type in any column
            prop_type_mask = subset.apply(lambda x: x.astype(str).str.contains(criteria['property_type'], case=False)).any(axis=1)
            subset = subset[prop_type_mask]
            st.write(f"After property type filter size: {len(subset)}")
        
        st.write("Final filtered table:")
        st.dataframe(subset)
        
        return subset


    async def load_and_query_investor(self, s3_client, bucket: str, investor_prefix: str, query: str, structured_criteria: dict):
        st.write(f"Processing investor {investor_prefix} with criteria:", structured_criteria)
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

                # Check if we have any relevant chunks before proceeding
                if not relevant_chunks:
                    st.write("No relevant chunks found")
                    return []

                text_content = "\n".join([chunk.page_content for chunk in relevant_chunks])
                
                # Analyze tables with structured criteria
                table_results = await self.analyze_tables(structured_criteria)
                st.write("Table analysis results:", table_results)

                if not table_results or isinstance(table_results, str) or table_results.get("error"):
                    st.write("No matching table results found")
                    return []

                # Create response if we have matches
                if table_results.get('matches', False):
                    # Safely get metadata from the first chunk
                    metadata = relevant_chunks[0].metadata if relevant_chunks else {}
                    
                    return [{
                        "name of investor": metadata.get("investor", "Unknown"),
                        "confidence": table_results.get('confidence_score', 0),
                        "details": text_content,
                        "restrictions": table_results.get('restrictions', []),
                        "credit score": table_results.get('min_credit_score', 0),
                        "loan to value": table_results.get('max_ltv', 0),
                        "footnotes": table_results.get('footnotes', []),
                        "source_url": metadata.get("s3_url", "")
                    }]

                return []

        except Exception as e:
            st.error(f"Error processing {investor_prefix}: {str(e)}")
            logging.error(f"Error processing {investor_prefix}: {str(e)}", exc_info=True)
            return []

    async def query_guidelines(self, query: str):
        
        # Parse query into structured criteria JSON
        structured_criteria_response = self.llm.invoke(
            self.query_parser_prompt.format(query=query)
        )
        
        # Clean up the JSON
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
            if hasattr(response, 'content'):
                # Clean up the content - remove markdown code blocks and extra whitespace
                content = response.content
                content = content.replace('```json', '').replace('```', '').strip()
                st.write("Cleaned content:", content)  # Debug
                return json.loads(content)
            return None
        except json.JSONDecodeError as e:
            st.write(f"JSON decode error: {e}")
            # If direct parsing fails, try to extract JSON with regex
            try:
                json_match = re.search(r'\{.*\}', str(response.content), re.DOTALL)
                if json_match:
                    extracted = json_match.group(0)
                    st.write("Extracted JSON:", extracted)  # Debug
                    return json.loads(extracted)
            except Exception as e:
                st.write(f"Regex extract error: {e}")
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