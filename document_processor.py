"""
Document processing module for the FinDoc Demo RAG application
"""
import streamlit as st
import time
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from utils import get_file_key, compute_file_hash, get_files_key, get_session_collection_name
from ui_components import render_file_analysis


class DocumentProcessor:
    """Processes documents and creates embeddings for the vector database"""
    
    def __init__(self, document_loader):
        self.document_loader = document_loader
        self.embedding_function = OpenAIEmbeddings()
    
    def process_file(self, user_file):
        """
        Processes an uploaded file and creates embeddings
        Returns retriever or None if processing fails
        """
        if user_file is None:
            return None
        
        # Check if file already processed
        current_file_key = get_file_key(user_file)
        if st.session_state.get('processed_file') == current_file_key:
            return st.session_state.get('retriever')
        
        try:
            return self._process_new_file(user_file, current_file_key)
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Please make sure your file is in a supported format and try again.")
            return None

    def process_files(self, uploaded_files):
        """Process multiple uploaded files and update the vector store.

        Returns a retriever over all indexed files for the current session.
        """
        if not uploaded_files:
            return None

        # Compute combined key based on file content hashes
        files_key = get_files_key(uploaded_files)
        if st.session_state.get('processed_files_key') == files_key and st.session_state.get('retriever') is not None:
            return st.session_state.get('retriever')

        # Build mapping: filename -> file_id (hash) for metadata tagging and diff
        file_id_by_name = {}
        for f in uploaded_files:
            file_id_by_name[f.name] = compute_file_hash(f)

        # Display simple summary (no persistent title)
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔄 Loading documents...")
            progress_bar.progress(15)
            documents = self.document_loader.load_multiple_uploaded_files(uploaded_files)

            status_text.text("✂️ Splitting into chunks...")
            progress_bar.progress(45)
            doc_splits = self._create_document_chunks(documents)

            # Attach file_id to metadata using original_filename from loader
            for split in doc_splits:
                original_name = split.metadata.get("original_filename")
                if original_name and original_name in file_id_by_name:
                    split.metadata["file_id"] = file_id_by_name[original_name]

            # Open/create session-scoped Chroma collection
            status_text.text("🧠 Updating vector store...")
            progress_bar.progress(70)
            chroma_db = self._get_chroma_store()

            # Determine additions and removals
            current_ids = set(file_id_by_name.values())
            previous_ids = set(st.session_state.get('processed_file_ids', set()))
            to_remove = list(previous_ids - current_ids)
            to_add = list(current_ids - previous_ids)

            # Remove docs for files no longer present
            for rid in to_remove:
                try:
                    chroma_db.delete(where={"file_id": rid})
                except Exception as _:
                    pass

            # Add docs for new files only
            if to_add:
                new_docs = [d for d in doc_splits if d.metadata.get("file_id") in set(to_add)]
                if new_docs:
                    chroma_db.add_documents(new_docs)

            # Ensure persisted
            try:
                chroma_db.persist()
            except Exception:
                pass

            # Finalize
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            retriever = chroma_db.as_retriever()

            # Update session state
            st.session_state.retriever = retriever
            st.session_state.processed_files_key = files_key
            st.session_state.processed_file_ids = current_ids

            # Clean UI
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            # Debug
            print(f"Indexed files: {len(current_ids)}; added {len(to_add)}, removed {len(to_remove)}")
            return retriever

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error processing files: {str(e)}")
            return None
    
    def _process_new_file(self, user_file, current_file_key):
        """Processes a new file that hasn't been processed before"""
        # Get file info and display analysis
        file_info = self.document_loader.get_upload_info(user_file)
        render_file_analysis(file_info)
        
        # Check if file type is supported
        if not file_info['is_supported']:
            st.error(f"❌ Unsupported file type: .{file_info['extension']}")
            st.info(f"📋 Supported formats: {self.document_loader.get_supported_extensions_display()}")
            return None
        
        # Process the file
        return self._execute_processing_pipeline(user_file, file_info, current_file_key)
    
    def _execute_processing_pipeline(self, user_file, file_info, current_file_key):
        """Runs the complete processing pipeline"""
        # Processing UI (no persistent title)
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load document
            status_text.text("🔄 Loading document...")
            progress_bar.progress(25)
            documents = self.document_loader.load_uploaded_file(user_file)
            
            # Step 2: Extract content
            status_text.text("🔍 Extracting content...")
            progress_bar.progress(50)
            st.success(f"✅ Successfully extracted content from {file_info['filename']}")
            
            # Step 3: Split into chunks
            progress_bar.progress(75)
            status_text.text("✂️ Splitting into chunks...")
            doc_splits = self._create_document_chunks(documents)
            
            # Step 4: Create embeddings
            progress_bar.progress(90)
            status_text.text("🧠 Creating embeddings...")
            chroma_db = self._create_vector_database(doc_splits)
            
            # Step 5: Complete
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Clean up UI
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Store in session state
            retriever = chroma_db.as_retriever()
            st.session_state.processed_file = current_file_key
            st.session_state.retriever = retriever
            
            # Debug: Confirm retriever creation and test it
            print(f"Retriever created successfully: {retriever is not None}")
            print(f"Session state updated with file key: {current_file_key}")
            
            # Test the retriever with a simple query
            try:
                test_docs = retriever.invoke("test")
                print(f"Retriever test successful - found {len(test_docs)} documents")
            except Exception as test_error:
                print(f"Retriever test failed: {test_error}")
            
            return retriever
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            raise e
    
    def _create_document_chunks(self, documents):
        """Splits documents into smaller chunks while preserving metadata"""
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        # Preserve metadata by splitting Document objects directly
        doc_splits = splitter.split_documents(documents)

        # Add chunk-level metadata
        total = len(doc_splits)
        for i, split in enumerate(doc_splits):
            split.metadata.update({
                "chunk_id": i,
                "total_chunks": total,
                "chunk_size": len(split.page_content),
            })
        return doc_splits
    
    def _create_vector_database(self, doc_splits):
        """Creates a ChromaDB vector database from document chunks"""
        # Legacy single-file path kept for backward compatibility
        return Chroma.from_documents(
            documents=doc_splits,
            collection_name=get_session_collection_name(CHROMA_COLLECTION_NAME),
            embedding=self.embedding_function,
            persist_directory=CHROMA_PERSIST_DIR
        )

    def _get_chroma_store(self):
        """Open or create a session-scoped Chroma collection to allow incremental add/delete."""
        collection_name = get_session_collection_name(CHROMA_COLLECTION_NAME)
        return Chroma(
            collection_name=collection_name,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embedding_function,
        )
