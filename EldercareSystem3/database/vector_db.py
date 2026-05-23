import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import cfg
from utils.logger import get_logger

logger = get_logger("vector_db")

class ReportVectorDB:
    """Vector database for storing LLM reports using ChromaDB."""
    
    def __init__(self):
        self.db_dir = cfg.output_dir / "database" / "chroma_db"
        self.client = None
        self.collection = None
        self.collection_name = "eldercare_reports"

    def clear_database(self):
        """Clear the existing database directory (for testing phase)."""
        if self.db_dir.exists():
            try:
                shutil.rmtree(self.db_dir, ignore_errors=True)
                logger.info(f"Cleared existing ChromaDB directory: {self.db_dir}")
            except Exception as e:
                logger.warning(f"Failed to clear ChromaDB directory: {e}")
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def load(self):
        if self.client is not None:
            return
            
        try:
            import chromadb
            logger.info("Initializing ChromaDB client...")
            self.client = chromadb.PersistentClient(path=str(self.db_dir))
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} # Default similarity
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None

    def add_report(self, clip_name: str, report_text: str, metadata: Dict[str, Any] = None):
        """Add a generated report text to the vector database."""
        if self.collection is None:
            logger.warning("ChromaDB is not loaded. Skipping add_report.")
            return
            
        if metadata is None:
            metadata = {}
            
        metadata["clip_name"] = clip_name
        
        try:
            self.collection.add(
                documents=[report_text],
                metadatas=[metadata],
                ids=[clip_name]  # Use clip_name as unique ID
            )
            logger.info(f"Report for '{clip_name}' added to Vector DB.")
        except Exception as e:
            logger.error(f"Failed to add report to Vector DB: {e}")

    def search(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search the database for similar reports."""
        if self.collection is None:
            logger.warning("ChromaDB is not loaded. Cannot search.")
            return []
            
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results and 'documents' in results and len(results['documents']) > 0:
                docs = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results.get('distances', [[0] * len(docs)])[0]
                
                for i in range(len(docs)):
                    formatted_results.append({
                        "document": docs[i],
                        "metadata": metadatas[i],
                        "distance": distances[i]
                    })
                    
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
