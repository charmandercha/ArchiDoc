import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox, 
    QTextEdit, QSplitter, QTreeView
)
from PyQt5.QtCore import Qt, QDir
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from ollama import chat
from ollama import ChatResponse
from extract_embedding import SemanticSearchChroma
# Importações do seu script original
from main_functions import (
    collect_python_files, 
    generate_documentation, 
    analyze_file, 
    save_documentation,
    log_info,
    log_error
)
searcher = SemanticSearchChroma()
class DocumentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Automated Project Documentation")
        self.setGeometry(100, 100, 1200, 800)
        self.localizacao_da_pasta = None
        # Main central widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Left Sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar.setLayout(sidebar_layout)
        sidebar.setMaximumWidth(300)
        
        # Project Directory Selection
        dir_label = QLabel("Select Project Directory:")
        self.dir_input = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.select_directory)
        
        self.partial_report_checkbox = QCheckBox("Partial Report")
        
        # Partial Report Criteria
        self.partial_criteria_input = QLineEdit()
        self.partial_criteria_input.setPlaceholderText("Enter partial report criteria")
        self.partial_criteria_input.setEnabled(False)
        
        # Toggle partial report input
        self.partial_report_checkbox.toggled.connect(
            self.partial_criteria_input.setEnabled
        )
        
        # Generate Report Button
        generate_button = QPushButton("Generate Documentation")
        generate_button.clicked.connect(self.generate_documentation)
        
        # Add widgets to sidebar
        sidebar_layout.addWidget(dir_label)
        sidebar_layout.addWidget(self.dir_input)
        sidebar_layout.addWidget(browse_button)
        sidebar_layout.addWidget(self.partial_report_checkbox)
        sidebar_layout.addWidget(self.partial_criteria_input)
        sidebar_layout.addWidget(generate_button)
        generate_individual_button = QPushButton("Generate Individual Reports")
        generate_individual_button.clicked.connect(self.generate_individual_reports)
        sidebar_layout.addWidget(generate_individual_button)
        # Crie o novo botão
        self.generate_embedding_button = QPushButton("Gerar embeddings")
        
        # self.generate_embedding_button.clicked.connect(searcher.add_documents(self.generate_embedding))
        self.generate_embedding_button.setEnabled(False)  # Desabilite o botão por padrão
        sidebar_layout.addWidget(self.generate_embedding_button) 
        sidebar_layout.addStretch(1)
        
        # Main Content Area
        content_area = QWidget()
        content_layout = QVBoxLayout()
        content_area.setLayout(content_layout)
        
        # Results Display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        # File Tree View
        self.file_tree = QTreeView()
        
        # Add widgets to content area
        content_layout.addWidget(QLabel("Documentation Results:"))
        content_layout.addWidget(self.results_text)
        content_layout.addWidget(self.file_tree)
        
        # Add components to main layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_area)
    def enable_new_button(self):
        self.generate_embedding_button.setEnabled(True)
        self.generate_embedding_button.clicked.connect(searcher.add_documents(self.localizacao_da_pasta))
    def generate_embedding(self):
            localizacao_da_pasta = self.localizacao_da_pasta  # obtém a localização da pasta
            if localizacao_da_pasta is not None:
                searcher.add_documents(localizacao_da_pasta)
            else:
                print("Localização da pasta não definida")     
    def select_directory(self):
        """Open directory selection dialog"""
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "Select Project Directory", 
            os.path.expanduser('~')
        )
        
        if dir_path:
            self.dir_input.setText(dir_path)
            self.populate_file_tree(dir_path)
    
    def populate_file_tree(self, directory):
        """Populate file tree with project structure"""
        model = QStandardItemModel()
        root_item = model.invisibleRootItem()
        
        try:
            log_info(f"Populating file tree for directory: {directory}")
            for root, dirs, files in os.walk(directory):
                dir_item = QStandardItem(os.path.basename(root))
                for file in files:
                    file_item = QStandardItem(file)
                    dir_item.appendRow(file_item)
                root_item.appendRow(dir_item)
            
            self.file_tree.setModel(model)
        except Exception as e:
            log_error(f"Error populating file tree: {e}")
    
    def generate_documentation(self):
        """Generate documentation based on user selections"""
        project_dir = self.dir_input.text()
        
        if not project_dir:
            self.results_text.setText("Please select a project directory")
            return
        
        try:
            # Collect Python files
            python_files = collect_python_files(project_dir)
            
            # Analyze files
            results = []
            for file in python_files:
                file_result = analyze_file(file)
                if file_result:
                    results.append(file_result)
            
            # Generate documentation
            documentation = generate_documentation(results)
            
            # Save documentation
            save_documentation(documentation)
            
            # Display results
            self.display_documentation_results(documentation)
        
        except Exception as e:
            log_error(f"Error generating documentation: {e}")
            self.results_text.setText(f"An error occurred: {str(e)}")

    

    def generate_file_report(self, file, file_result):
        # Crie um prompt para a LLM
        prompt = f"""
        Extraia as informações mais relevantes do arquivo {file} e gere um texto conciso e informativo que descreva seu conteúdo. O texto gerado deve ser otimizado para busca semântica, utilizando embeddings.

        Exemplo de saída desejada:

        Tópicos principais: mudanças climáticas, agricultura, impacto ambiental, segurança alimentar, análise de dados, modelagem estatística.
        Conteúdo: Este estudo científico investiga os efeitos das mudanças climáticas na produção agrícola global. Através da análise de dados históricos e projeções futuras, o documento demonstra como eventos climáticos extremos, como secas e inundações, afetam a produtividade agrícola e a disponibilidade de alimentos. Os autores propõem medidas de adaptação e mitigação para garantir a segurança alimentar em um cenário de aquecimento global.

        """

        # Chame a LLM
        response: ChatResponse = chat(model='qwen2.5:14b-instruct-q4_K_M', messages=[
            {
                'role':'system',
                'content': 'You are an expert in code analysis.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ])
        
        # Retorne o relatório gerado pela LLM
        return response.message.content

    def generate_individual_reports(self):
        project_dir = self.dir_input.text()
        if not project_dir:
            self.results_text.setText("Por favor, selecione um diretório de projeto")
            return
        
        try:
            # Collect Python files
            python_files = collect_python_files(project_dir)
            
            # Crie uma pasta separada para armazenar os relatórios individuais
            reports_dir = os.path.join(project_dir, "_relatorios")
            self.localizacao_da_pasta = reports_dir
            os.makedirs(reports_dir, exist_ok=True)
            
            # Gere relatórios individuais para cada arquivo
            for file in python_files:
                file_result = analyze_file(file)
                if file_result:
                    # Gere um relatório para o arquivo
                    report = self.generate_file_report(file, file_result)
                    # Salve o relatório em uma pasta separada
                    report_file = os.path.join(reports_dir, f"{os.path.basename(file)}.txt")
                    with open(report_file, "w") as f:
                        f.write(report)
            
            self.results_text.setText("Relatórios individuais gerados com sucesso!")
            self.enable_new_button()
        except Exception as e:
            log_error(f"Erro ao gerar relatórios individuais: {e}")
            self.results_text.setText(f"Ocorreu um erro: {str(e)}")
     
    def display_documentation_results(self, documentation):
        """Display documentation results in the text area"""
        results_text = f"""
Project Overview:
{documentation['project_overview']}

Module Interactions:
{documentation['module_interactions']}

File Summaries:
"""
        for file_path, file_info in documentation['file_summaries'].items():
            results_text += f"\n{file_path}:\n{file_info['summary']}\n"
        
        self.results_text.setText(results_text)

def main():
    app = QApplication(sys.argv)
    main_window = DocumentationApp()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()