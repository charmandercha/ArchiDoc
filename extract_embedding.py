import os
import ollama
import chromadb

class SemanticSearchChroma:
    def __init__(self, collection_name='document_embeddings'):
        """
        Inicializa cliente ChromaDB e cria/carrega coleção
        """
        self.client = chromadb.PersistentClient(path="./chroma_storage")
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, directory):
        """
        Adiciona documentos do diretório à coleção ChromaDB
        """
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Gera embedding
                embedding = ollama.embeddings(
                    model='mxbai-embed-large', 
                    prompt=content
                )['embedding']
                
                # Adiciona ao ChromaDB
                self.collection.add(
                    embeddings=[embedding],
                    documents=[content],
                    ids=[filename]
                )

    def search(self, query, n_results=3):
        """
        Busca semântica com ChromaDB
        """
        query_embedding = ollama.embeddings(
            model='mxbai-embed-large', 
            prompt=query
        )['embedding']
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results

def main():
    base_dir = '/home/marcos/projetos_automatizacao/meu_primeiro_agent/_relatorios/'
    
    # Inicializa e popula
    searcher = SemanticSearchChroma()
    searcher.add_documents(base_dir)
    
    # Exemplo de busca
    query = input("Digite o assunto para busca: ")
    results = searcher.search(query)
    
    # Exibe resultados
    for i, doc in enumerate(results['documents'][0], 1):
        print(f"Resultado {i}:\n{doc[:500]}...\n")

if __name__ == '__main__':
    main()