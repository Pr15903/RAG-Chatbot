# MVC Project with Python API with FAISS Vector DB and LLM Integration

## Overview
This project is an **MVC-based web application** integrated with a **Python API** to handle file uploads, process embeddings using `all-MiniLM-L6-v2`, store them in **FAISS vector database**, and retrieve relevant documents for a **Phi-3 LLM** chatbot. The chatbot maintains conversation history to provide context-aware responses.

## Features
- **File Upload & Deletion**: Users can select multiple files to upload and delete via the MVC UI.
- **Embeddings Storage**: Uploaded files are processed using `all-MiniLM-L6-v2` and stored in **FAISS**.
- **Conversational Memory**: Chatbot maintains history using `LangChain ChatPromptTemplate`.
- **Query Processing**: On user input, the system retrieves relevant documents and passes them to **Phi-3 LLM** for context-aware responses.

## Tech Stack
- **Frontend (MVC)**: ASP.NET MVC
- **Backend (Python API)**: Flask / FastAPI
- **Embeddings Model**: `all-MiniLM-L6-v2`
- **Vector Database**: FAISS
- **LLM Model**: Phi-3
- **Chat Framework**: LangChain

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- .NET 5+
- Virtual environment (optional but recommended)

### Backend Setup (Python API)
1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd <repo-name>
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the API:
   ```sh
   python Chatbot.py  # Adjust the filename if using FastAPI
   ```

### Frontend Setup (MVC Application)
1. Open the MVC project in Visual Studio.
2. Restore dependencies:
   ```sh
   dotnet restore
   ```
3. Run the MVC application.

## Usage
1. **Upload Files**: Navigate to the UI and select multiple files to upload.
2. **Processing**: Files are sent to the Python API, embeddings are created, and stored in FAISS.
3. **Query LLM**: Ask questions in the chat UI; the system retrieves related documents and generates responses using Phi-3.
4. **Chat History**: Previous messages are used for better contextual answers.
5. **Delete Files**: Remove stored files using the UI option.

## API Endpoints
- **`POST /upload`**: Uploads and processes files.
- **`POST /query`**: Sends query to Phi-3 LLM and returns a response.
- **`DELETE /delete/<file_id>`**: Deletes a stored file.(remaining)

## Future Enhancements
- UI improvements for better file management.
- Support for additional embedding models.
- Enhanced logging and error handling.
- Multi-user chat session support.

## Contributing
1. Fork the repository.
2. Create a feature branch.
3. Commit changes and push to your fork.
4. Open a Pull Request.

## License
This project is licensed under the MIT License.

## Contact
For any issues or feature requests, please raise a GitHub issue or reach out to the project maintainer.

