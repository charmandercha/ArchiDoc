import os
import ast
import json
import time
import numpy as np
from typing import List, Dict
import ollama
from ollama import chat, ChatResponse
import colorama
from tqdm import tqdm
import markdown

# Configurações existentes mantidas
colorama.init(autoreset=True)

# Adicionando modelo de embeddings

def log_info(message):
    """Prints informative messages in blue"""
    print(f"{colorama.Fore.CYAN}[INFO] {message}{colorama.Fore.RESET}")

def log_warning(message):
    """Prints warning messages in yellow"""
    print(f"{colorama.Fore.YELLOW}[WARN] {message}{colorama.Fore.RESET}")

def log_error(message):
    """Prints error messages in red"""
    print(f"{colorama.Fore.RED}[ERROR] {message}{colorama.Fore.RESET}")

def log_success(message):
    """Prints success messages in green"""
    print(f"{colorama.Fore.GREEN}[SUCCESS] {message}{colorama.Fore.RESET}")

def analyze_file(file_path: str) -> Dict:
    """Analyzes an individual Python file with enhanced metadata"""
    try:
        log_info(f"Analyzing file: {file_path}")
        start_time = time.time()
        
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
            tree = ast.parse(code)
            file_info = {
                "file": file_path,
                "filename": os.path.basename(file_path),
                "classes": [],
                "functions": [],
                "imports": [],
                "docstrings": [],
                "complexity": 0,
                "summary": ""  # Placeholder for AI-generated summary
            }
        
        # Existing AST analysis logic
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                file_info["classes"].append({
                    "name": node.name,
                    "methods": [method.name for method in node.body if isinstance(method, ast.FunctionDef)]
                })
            elif isinstance(node, ast.FunctionDef):
                file_info["functions"].append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "complexity": len(list(ast.walk(node)))
                })
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                file_info["imports"].append(ast.unparse(node))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                file_info["docstrings"].append(node.value.s)
        
        # AI-generated summary (can be enhanced later)
        file_info["summary"] = generate_file_summary(file_info)
        
        end_time = time.time()
        log_success(f"File analysis completed in {end_time - start_time:.2f} seconds")
        return file_info
    except Exception as e:
        log_error(f"Error analyzing {file_path}: {e}")
        return {}

def generate_file_summary(file_info: Dict) -> str:
    """Generate a basic summary for a file"""
    summary = f"Arquivo: {file_info['filename']}\n"
    summary += f"Classes: {', '.join([cls['name'] for cls in file_info['classes']] or ['Nenhuma'])}\n"
    summary += f"Funções: {', '.join([func['name'] for func in file_info['functions']] or ['Nenhuma'])}\n"
    return summary

def collect_python_files(directory: str) -> List[str]:
    """Collects all Python files in a directory"""
    log_info(f"Collecting Python files in: {directory}")
    python_files = []
    total_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
                total_files += 1
    
    log_success(f"Found {total_files} Python files")
    return python_files

def generate_embeddings(descriptions: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Generate embeddings for file descriptions"""
    log_info("Generating semantic embeddings")
    embeddings = {}
    
    for file_path, description in descriptions.items():
        embedding = ollama.embeddings(
            model='mxbai-embed-large',
            prompt=description
        )
        embeddings[file_path] = embedding
        
        # Opcional: salvar embeddings
        embedding_dir = os.path.join(os.path.dirname(file_path), '.project_docs')
        os.makedirs(embedding_dir, exist_ok=True)
        np.save(os.path.join(embedding_dir, f"{os.path.basename(file_path)}.embedding.npy"), embedding)
    
    log_success(f"Generated {len(embeddings)} embeddings")
    return embeddings

def search_project_files(project_dir: str, query: str, top_k: int = 5) -> List[tuple]:
    """Semantic search across project files"""
    log_info(f"Performing semantic search for query: {query}")
    query_embedding = ollama.embeddings(
        model='mxbai-embed-large',
        prompt=query
    )
    similarities = {}
    
    # Buscar arquivos e embeddings
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py'):
                embedding_path = os.path.join(root, '.project_docs', f"{file}.embedding.npy")
                if os.path.exists(embedding_path):
                    file_embedding = np.load(embedding_path)
                    # Calcula a similaridade entre os embeddings
                    # A forma como você calcula a similaridade pode variar dependendo do tipo de embedding
                    similarity = np.dot(query_embedding, file_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(file_embedding)
                    )
                    similarities[os.path.join(root, file)] = similarity
    
    # Ordenar e retornar top k resultados
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    log_success(f"Found {len(sorted_results)} relevant files")
    return sorted_results

def generate_documentation(analysis_results: List[Dict], model: str = "qwen2.5:14b-instruct-q4_K_M") -> Dict:
    """Generates documentation using LLM"""
    documentation = {
        "project_overview": "",
        "file_summaries": {},
        "module_interactions": ""
    }
    
    # Project overview analysis
    log_info("Generating project overview")
    overview_prompt = "Analyze this project structure and provide a comprehensive overview:\n\n"
    for result in analysis_results:
        overview_prompt += f"File: {result['file']}\n"
        overview_prompt += f"Classes: {', '.join([cls['name'] for cls in result['classes']] or ['Nenhuma'])}\n"
        overview_prompt += f"Functions: {', '.join([func['name'] for func in result['functions']] or ['Nenhuma'])}\n\n"
    
    overview_prompt += "Describe the project's purpose, main components, and how they interact."
    
    try:
        response: ChatResponse = chat(model=model, messages=[
            {'role':'system', 'content': 'You are an expert in machine learning project analysis.'},
            {'role': 'user', 'content': overview_prompt}
        ])
        documentation["project_overview"] = response.message.content
        log_success("Project overview generated")
    except Exception as e:
        log_error(f"Error generating project overview: {e}")
    
    # Individual file summaries
    log_info("Generating file summaries")
    for result in tqdm(analysis_results, desc="Processing files"):
        try:
            file_summary_prompt = f"Analyze the file {result['file']} and explain its purpose and key components:\n"
            file_summary_prompt += f"Classes: {', '.join([cls['name'] for cls in result['classes']] or ['Nenhuma'])}\n"
            file_summary_prompt += f"Functions: {', '.join([func['name'] for func in result['functions']] or ['Nenhuma'])}\n"
            
            response: ChatResponse = chat(model=model, messages=[
                {'role':'system', 'content': 'You are an expert in code analysis.'},
                {'role': 'user', 'content': file_summary_prompt}
            ])
            
            documentation["file_summaries"][result['file']] = {
                "summary": response.message.content,
                "details": result
            }
        except Exception as e:
            log_warning(f"Error generating summary for {result['file']}: {e}")
    
    # Module interactions
    log_info("Generating module interaction description")
    try:
        interaction_prompt = "Describe how the modules and components in this project interact with each other."
        response: ChatResponse = chat(model=model, messages=[
            {'role':'system', 'content': 'You are an expert in software architecture.'},
            {'role': 'user', 'content': interaction_prompt}
        ])
        documentation["module_interactions"] = response.message.content
        log_success("Module interaction description generated")
    except Exception as e:
        log_error(f"Error generating module interactions: {e}")
    
    return documentation

def save_documentation(documentation: Dict, output_dir: str = "project_docs"):
    """Saves documentation in multiple formats"""
    log_info(f"Saving documentation to directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, "project_documentation.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(documentation, f, indent=2)
        log_success(f"JSON documentation saved to: {json_path}")
    except Exception as e:
        log_error(f"Error saving JSON: {e}")
    
    # Generate Markdown
    markdown_content = f"""# Project Documentation

## Project Overview
{documentation['project_overview']}

## Module Interactions
{documentation['module_interactions']}

## File Summaries
"""
    
    for file_path, file_info in documentation['file_summaries'].items():
        markdown_content += f"""
### {file_path}
{file_info['summary']}

#### Detailed Components
- **Classes**: {', '.join([cls['name'] for cls in file_info['details']['classes']])}
- **Functions**: {', '.join([func['name'] for func in file_info['details']['functions']])}
"""
    
    # Save Markdown
    md_path = os.path.join(output_dir, "project_documentation.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        log_success(f"Markdown documentation saved to: {md_path}")
    except Exception as e:
        log_error(f"Error saving Markdown: {e}")
    
    # Convert to HTML
    try:
        html_content = markdown.markdown(markdown_content)
        html_path = os.path.join(output_dir, "project_documentation.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Project Documentation</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """)
        log_success(f"HTML documentation saved to: {html_path}")
    except Exception as e:
        log_error(f"Error saving HTML: {e}")

if __name__ == "__main__":
    # Start of execution
    start_total_time = time.time()
    log_info("Starting project analysis")
    
    # Project directory
    project_dir = "/home/marcos/projetos_automatizacao/ENTENDER_textgrad/textgrad"
    
    # File collection
    python_files = collect_python_files(project_dir)
    
    # File analysis
    log_info("Starting detailed file analysis")
    results = []
    for file in tqdm(python_files, desc="Analyzing files"):
        print(file)
        file_result = analyze_file(file)
        if file_result:
            results.append(file_result)
    
    # Documentation generation
    log_info("Generating documentation with AI assistant")
    documentation = generate_documentation(results)
    
    # Saving documentation
    save_documentation(documentation)
    
    # Total execution time
    end_total_time = time.time()
    log_success(f"Analysis completed in {end_total_time - start_total_time:.2f} seconds")
    print("\nDocumentation generated successfully in 'project_docs' directory!")