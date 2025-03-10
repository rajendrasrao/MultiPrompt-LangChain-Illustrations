---
title: MultiPrompt with LangChain
emoji: 🐨
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.42.0
app_file: multi_prompt.py
pinned: false
license: apache-2.0
short_description: Refined langgraphAgenticAI
---

### End To End Agentic AI Projects


**MultiPrompt with LangChain**

This repository demonstrates how to implement a MultiPrompt setup using LangChain and Streamlit. With this setup, users can add multiple prompt experts through a Streamlit UI and query them to receive responses based on the most relevant prompt.

**Features**
**Multiple Prompt Experts**: Define and manage multiple prompts, each tailored to specific types of queries.

**Streamlit Integration**: A user-friendly web interface to add, edit, and manage prompts dynamically.

**Query Resolution**: Ask questions, and LangChain determines the most relevant prompt expert to provide a response.

**How It Works**

**Prompt Management**

Use the Streamlit UI to add prompt experts.

Each prompt expert can specialize in a particular topic or domain.

**Query Handling**

When a user asks a question, LangChain uses a relevance mechanism to select the most suitable prompt expert.

The selected expert generates the response.

**Setup Instructions**
**Prerequisites**

Python 3.8 or higher

Streamlit installed

LangChain library installed

**Installation**

**Clone the repository:**
**git clone**

git clone https://github.com/rajendrasrao/MultiPrompt-LangChain-Illustrations.git
cd MultiPrompt-LangChain-Illustrations

**Install dependencies**:

pip install -r requirements.txt

**.env file setup**:
please create .env file and set below env var for google gemeni APi key.

GOOGLE_API_KEY=your gemeni api key (.env file is added in .gitignore file)

**Run the Application**

Start the Streamlit app:

streamlit run init_add_prompt.py

Open your browser and navigate to http://localhost:8501 to interact with the UI.

**Usage**

Adding Prompt Experts



Use the "Add Prompt Expert" section in the Streamlit app to define a new prompt.

Specify the prompt's name, description, and complete the prompt to categorize it.

**Asking Questions**

Run the following command to enable querying:

streamlit run multi_prompt.py

Enter your query in the text box provided on the page.

The system evaluates the query and determines the most relevant prompt expert.

View the response generated by the selected expert.





**Customization**

You can extend this project by:

Adding more complex selection logic for prompts.

Enabling persistent storage for prompt experts.

Supporting more advanced LangChain capabilities.

**License**

This project is licensed under the MIT License. See the LICENSE file for details.

**Contributions**

Contributions are welcome! Feel free to open issues or submit pull requests.

