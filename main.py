import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox,
    QTextEdit, QSplitter, QTreeView, QMenu, QAction
)
from PyQt5.QtCore import Qt, QDir, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from ollama import chat
from ollama import ChatResponse
from extract_embedding import SemanticSearchChroma
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

        # Generate Individual Reports Button
        generate_individual_button = QPushButton("Generate Individual Reports")
        generate_individual_button.clicked.connect(self.generate_individual_reports)
        sidebar_layout.addWidget(generate_individual_button)

        # Generate Embeddings Button
        self.generate_embedding_button = QPushButton("Gerar embeddings")
        self.generate_embedding_button.setEnabled(False)
        self.generate_embedding_button.clicked.connect(self.generate_embedding)
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
        self.file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self.show_context_menu)

        # Add widgets to content area
        content_layout.addWidget(QLabel("Documentation Results:"))
        content_layout.addWidget(self.results_text)
        content_layout.addWidget(self.file_tree)

        # Add components to main layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_area)

    def enable_new_button(self):
        self.generate_embedding_button.setEnabled(True)

    def generate_embedding(self):
        if self.localizacao_da_pasta is not None:
            searcher.add_documents(self.localizacao_da_pasta)
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
                # Criar um item para o diretório atual
                dir_item = QStandardItem(os.path.basename(root))
                dir_item.setData(root, Qt.UserRole + 1)  # Armazena o caminho completo do diretório
                
                for file in files:
                    file_item = QStandardItem(file)
                    # Armazena o caminho completo do arquivo
                    full_file_path = os.path.join(root, file)
                    file_item.setData(full_file_path, Qt.UserRole + 1)
                    dir_item.appendRow(file_item)
                
                root_item.appendRow(dir_item)
            
            self.file_tree.setModel(model)
            self.file_tree.selectionModel().selectionChanged.connect(self.generate_individual_report)
        except Exception as e:
            log_error(f"Error populating file tree: {e}")

    def generate_individual_report(self, selected, deselected):
        """Generate individual report for a selected file"""
        selected_indexes = self.file_tree.selectedIndexes()
        if selected_indexes:
            # Recupera o item selecionado
            selected_item = selected_indexes[0].model().itemFromIndex(selected_indexes[0])
            
            # Recupera o caminho completo do arquivo
            file_path = selected_item.data(Qt.UserRole + 1)
            
            # Verifica se é um arquivo (não um diretório)
            if os.path.isfile(file_path):
                print(f"Arquivo selecionado: {file_path}")
                
                try:
                    file_result = analyze_file(file_path)
                    if file_result:
                        report = self.generate_file_report(file_path, file_result)
                        self.results_text.setText(report)
                except Exception as e:
                    log_error(f"Erro ao gerar relatório para {file_path}: {e}")
                    self.results_text.setText(f"Erro ao analisar arquivo: {e}")
    def generate_file_report(self, file, file_result):
        # Create a prompt for the LLM
        prompt = f"""
        Extract the most relevant information from the file {file} and generate a concise and informative text describing its content. 
        The generated text should be optimized for semantic search using embeddings.

        Desired output example:

        Main topics: climate change, agriculture, environmental impact, food security, data analysis, statistical modeling.
        Content: This scientific study investigates the effects of climate change on global agricultural production. 
        By analyzing historical data and future projections, the document demonstrates how extreme climate events, 
        such as droughts and floods, affect agricultural productivity and food availability. 
        The authors propose adaptation and mitigation measures to ensure food security in a global warming scenario.
        """

        # Call the LLM
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
        
        # Return the report generated by the LLM
        return response.message.content

    def generate_documentation(self):
        project_dir = self.dir_input.text()
        if not project_dir:
            self.results_text.setText("Please select a project directory")
            return
        
        try:
            documentation = generate_documentation(project_dir)
            self.display_documentation_results(documentation)
        except Exception as e:
            log_error(f"Error generating documentation: {e}")
            self.results_text.setText(f"An error occurred: {str(e)}")

    def generate_individual_reports(self):
        project_dir = self.dir_input.text()
        if not project_dir:
            self.results_text.setText("Please select a project directory")
            return
        
        try:
            # Collect Python files
            python_files = collect_python_files(project_dir)
            
            # Create a separate folder to store individual reports
            reports_dir = os.path.join(project_dir, "_relatorios")
            self.localizacao_da_pasta = reports_dir
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate individual reports for each file
            for file in python_files:
                file_result = analyze_file(file)
                if file_result:
                    # Generate a report for the file
                    report = self.generate_file_report(file, file_result)
                    
                    # Save the report in a separate folder
                    report_file = os.path.join(reports_dir, f"{os.path.basename(file)}.txt")
                    with open(report_file, "w") as f:
                        f.write(report)
            
            self.results_text.setText("Individual reports generated successfully!")
            self.enable_new_button()
        except Exception as e:
            log_error(f"Error generating individual reports: {e}")
            self.results_text.setText(f"An error occurred: {str(e)}")

    def show_context_menu(self, pos):
        """Show context menu for selected file in the tree view"""
        selected_indexes = self.file_tree.selectedIndexes()
        if selected_indexes:
            selected_file = selected_indexes[0].data()
            context_menu = QMenu()
            generate_report_action = QAction(f"Generate Report for {selected_file}", self)
            generate_report_action.triggered.connect(lambda: self.generate_individual_report(None, None))
            context_menu.addAction(generate_report_action)
            context_menu.exec_(self.file_tree.mapToGlobal(pos))

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