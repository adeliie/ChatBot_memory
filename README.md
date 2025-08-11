# ChatBot Memory

A modular Python chatbot with document retrieval, memory, and user context management. Built with LangChain, ChromaDB, and HuggingFace embeddings.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/adeliie/ChatBot_memory.git
   cd ChatBot_memory
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt


3. **Configure your API key**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your real key:
     ```
     MEM0_API_KEY=your_real_key_here
     ```

## Usage

1. **Run the chatbot**
   ```bash
   python main.py
   ```

## Project Structure

- `main.py`: entry point
- `config.py`: configuration, embeddings, API key loading
- `memory_db.py`: memory and document database management
- `chatbot.py`: conversational logic
- `utils.py`: utility functions
- `core/`, `src/`: specialized modules
- `.env.example`: config template (never put your real key on GitHub)




