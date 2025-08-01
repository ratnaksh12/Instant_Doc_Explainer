Instant Doc Explainer

An AI-powered conversational application that allows users to upload PDF, DOCX, and TXT files and ask questions about their content using a Retrieval-Augmented Generation (RAG) system.

✨ Features
Multi-File Support: Processes and analyzes multiple files at once, including .pdf, .docx, and .txt formats.

Conversational Q&A: Engage in a natural language conversation to retrieve specific information and insights from your documents.

Vector Search: Utilizes a FAISS vector store for efficient semantic searching of document content.

Streamlit UI: Provides a simple, clean, and intuitive user interface built with the Streamlit framework.

🚀 Technologies Used
Frontend: Streamlit

Backend: LangChain, Groq API (gemma2-9b-it model)

Embeddings: HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)

Vector Store: FAISS

Dependencies: pypdf, python-docx, groq

⚙️ Setup and Installation
1. Clone the Repository
First, clone this repository to your local machine using Git.

git clone https://github.com/ratnaksh12/Instant-Doc-Explainer.git
cd Instant-Doc-Explainer

2. Set Up a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install Dependencies
Install all the required Python packages from the requirements.txt file.

pip install -r requirements.txt

4. Configure Your Groq API Key
This application requires an API key from Groq. For security, you must set this as an environment variable.

Go to the Groq console and generate an API key.

Set the GROQ_API_KEY environment variable on your system.

For Windows (Command Prompt):

set GROQ_API_KEY="YOUR_API_KEY"

For macOS/Linux (Terminal):

export GROQ_API_KEY="YOUR_API_KEY"

Note: For a more permanent solution, add this line to your shell's profile file (e.g., .bashrc, .zshrc).

5. Run the Application
Start the Streamlit application from your terminal.

streamlit run streamlit_app.py

The application will open in your web browser. You can now upload your documents and start asking questions!

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
