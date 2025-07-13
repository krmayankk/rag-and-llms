from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict

# Load your OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SimpleRAG:
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_document(self, title: str, content: str):
        """Add a document to our knowledge base"""
        print(f"Adding document: {title}")
        
        # Create embedding for this document
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=content
        )
        
        embedding = response.data[0].embedding
        
        # Store document and its embedding
        self.documents.append({
            'title': title,
            'content': content
        })
        self.embeddings.append(embedding)
        
        print(f"✅ Added '{title}' to knowledge base")
    
    def search_documents(self, query: str, top_k: int = 2) -> List[Dict]:
        """Search for relevant documents"""
        if not self.documents:
            return []
        
        print(f"🔍 Searching for: '{query}'")
        
        # Get embedding for the query
        query_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_embedding = query_response.data[0].embedding
        
        # Calculate similarity with all documents
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((i, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, similarity in similarities[:top_k]:
            results.append({
                'document': self.documents[i],
                'similarity': similarity
            })
            print(f"📄 Found relevant doc: '{self.documents[i]['title']}' (similarity: {similarity:.3f})")
        
        return results
    
    def ask_question(self, question: str) -> str:
        """Ask a question and get an answer based on our documents"""
        print(f"\n❓ Question: {question}")
        
        # Find relevant documents
        relevant_docs = self.search_documents(question)
        
        if not relevant_docs:
            return "I don't have any relevant documents to answer this question."
        
        # Build context from relevant documents
        context = ""
        for result in relevant_docs:
            doc = result['document']
            context += f"Document: {doc['title']}\nContent: {doc['content']}\n\n"
        
        # Generate answer using GPT
        print("🤖 Generating answer...")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Answer the user's question based ONLY on the provided documents. If the documents don't contain relevant information, say so."
                },
                {
                    "role": "user", 
                    "content": f"Context from documents:\n{context}\n\nQuestion: {question}\n\nAnswer based on the provided context:"
                }
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        print(f"✨ Answer: {answer}")
        return answer

# Example usage
def main():
    # Create our RAG system
    rag = SimpleRAG()
    
    # Add some sample company documents
    rag.add_document(
        "Remote Work Policy", 
        "Employees can work remotely up to 3 days per week with manager approval. "
        "A home office stipend of $500 annually is available for equipment purchases. "
        "All remote workers must attend monthly in-person team meetings."
    )
    
    rag.add_document(
        "Vacation Policy",
        "New employees receive 15 days of PTO annually. After 3 years of employment, "
        "this increases to 20 days. After 5 years, employees receive 25 days of PTO. "
        "Unused PTO can be carried over to the next year, up to a maximum of 5 days."
    )
    
    rag.add_document(
        "Expense Reimbursement",
        "All expense reports must be submitted within 30 days of the expense date. "
        "Receipts are required for all expenses over $25. Business meals are reimbursed "
        "up to $75 per day for client meetings. Personal alcohol expenses are not reimbursable."
    )
    
    print("\n" + "="*50)
    print("🎉 RAG System Ready! Ask me about company policies.")
    print("="*50)
    
    # Ask some questions
    questions = [
        "How many vacation days do new employees get?",
        "Can I work from home?",
        "What's the limit for meal expenses?",
        "Can I expense wine with a client dinner?"
    ]
    
    for question in questions:
        print("\n" + "-"*50)
        answer = rag.ask_question(question)
        print("-"*50)

if __name__ == "__main__":
    main()
