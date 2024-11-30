def save_vector_store_to_s3(self, investor_name: str) -> bool:
    """Save vector store to S3"""
    if not self.vector_store:
        return False
        
    try:
        # Create a temporary directory to save the vector store
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = os.path.join(temp_dir, investor_name)
            
            # Save vector store locally first
            self.vector_store.save_local(local_path)
            
            # Upload all files in the directory to S3
            for filename in os.listdir(local_path):
                file_path = os.path.join(local_path, filename)
                s3_key = f"vector_stores/{investor_name}/{filename}"
                
                with open(file_path, 'rb') as f:
                    self.s3_client.upload_fileobj(
                        f, 
                        BUCKET_NAME,
                        s3_key
                    )
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to save vector store to S3: {str(e)}")
        return False

def load_vector_store_from_s3(self, investor_name: str) -> bool:
    """Load vector store from S3"""
    try:
        # Create a temporary directory to download the vector store
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = os.path.join(temp_dir, investor_name)
            os.makedirs(local_path)
            
            # List all files for this investor's vector store
            prefix = f"vector_stores/{investor_name}/"
            response = self.s3_client.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return False
                
            # Download all files
            for obj in response['Contents']:
                filename = os.path.basename(obj['Key'])
                local_file_path = os.path.join(local_path, filename)
                
                self.s3_client.download_file(
                    BUCKET_NAME,
                    obj['Key'],
                    local_file_path
                )
            
            # Load the vector store
            loaded_vectorstore = FAISS.load_local(
                local_path,
                self.embeddings
            )
            
            if self.vector_store is None:
                self.vector_store = loaded_vectorstore
            else:
                self.vector_store.merge_from(loaded_vectorstore)
                
            return True
            
    except Exception as e:
        logging.error(f"Failed to load vector store from S3: {str(e)}")
        return False

def load_all_vector_stores_from_s3(self):
    """Load all available vector stores from S3"""
    try:
        # List all directories in the vector_stores prefix
        response = self.s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix='vector_stores/',
            Delimiter='/'
        )
        
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                # Extract investor name from prefix
                investor_name = prefix['Prefix'].split('/')[-2]
                self.load_vector_store_from_s3(investor_name)
                
    except Exception as e:
        logging.error(f"Failed to list vector stores in S3: {str(e)}")

# Modify the existing methods to use S3 storage
def load_and_process_pdf(self, uploaded_file):
    """Process single uploaded PDF and save vectors to S3"""
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
        
        # Upload PDF to S3
        s3_url = self.upload_to_s3(uploaded_file, uploaded_file.name)
        if s3_url:
            for chunk in chunks:
                chunk.metadata["s3_url"] = s3_url
        
        # Create/update vector store
        if self.vector_store:
            self.vector_store.add_documents(chunks)
        else:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Save vector store to S3
        self.save_vector_store_to_s3(investor)
        
        os.unlink(tmp_file.name)
        return investor, s3_url

def __init__(self, openai_api_key: str):
    self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    self.llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        temperature=0,
        openai_api_key=openai_api_key
    )
    self.vector_store = None
    self.s3_client = boto3.client(
        's3',
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )
    
    # Load existing vector stores at initialization
    self.load_all_vector_stores_from_s3()
    
    # Rest of the initialization code...