import os
import ast
import json
import markdown
import time
from typing import List, Dict
from openai import OpenAI
import colorama
from tqdm import tqdm

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Ollama client configuration
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

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
    """Analyzes an individual Python file"""
    try:
        log_info(f"Analyzing file: {file_path}")
        start_time = time.time()
        
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
            tree = ast.parse(code)
            file_info = {
                "file": file_path,
                "classes": [],
                "functions": [],
                "imports": [],
                "docstrings": [],
                "complexity": 0
            }
        
        # Detailed AST analysis
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
        
        end_time = time.time()
        log_success(f"File analysis completed in {end_time - start_time:.2f} seconds")
        return file_info
    except Exception as e:
        log_error(f"Error analyzing {file_path}: {e}")
        return {}

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
        overview_prompt += f"Classes: {', '.join([cls['name'] for cls in result['classes']])}\n"
        overview_prompt += f"Functions: {', '.join([func['name'] for func in result['functions']])}\n\n"
    
    overview_prompt += "Describe the project's purpose, main components, and how they interact."
    
    try:
        overview_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in machine learning project analysis."},
                {"role": "user", "content": overview_prompt}
            ]
        )
        documentation["project_overview"] = overview_completion.choices[0].message.content
        log_success("Project overview generated")
    except Exception as e:
        log_error(f"Error generating project overview: {e}")
    
    # Individual file summaries
    log_info("Generating file summaries")
    for result in tqdm(analysis_results, desc="Processing files"):
        try:
            file_summary_prompt = f"Analyze the file {result['file']} and explain its purpose and key components:\n"
            file_summary_prompt += f"Classes: {', '.join([cls['name'] for cls in result['classes']])}\n"
            file_summary_prompt += f"Functions: {', '.join([func['name'] for func in result['functions']])}\n"
            
            file_summary_completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in code analysis."},
                    {"role": "user", "content": file_summary_prompt}
                ]
            )
            
            documentation["file_summaries"][result['file']] = {
                "summary": file_summary_completion.choices[0].message.content,
                "details": result
            }
        except Exception as e:
            log_warning(f"Error generating summary for {result['file']}: {e}")
    
    # Module interactions
    log_info("Generating module interaction description")
    try:
        interaction_prompt = "Describe how the modules and components in this project interact with each other."
        interaction_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in software architecture."},
                {"role": "user", "content": interaction_prompt}
            ]
        )
        documentation["module_interactions"] = interaction_completion.choices[0].message.content
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
- **Classes**: {[cls['name'] for cls in file_info['details']['classes']]}
- **Functions**: {[func['name'] for func in file_info['details']['functions']]}
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
    project_dir = "./zou-group-textgrad"
    
    # File collection
    python_files = collect_python_files(project_dir)
    
    # File analysis
    log_info("Starting detailed file analysis")
    results = []
    for file in tqdm(python_files, desc="Analyzing files"):
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