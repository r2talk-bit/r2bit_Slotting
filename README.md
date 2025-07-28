# R2Bit Slotting Manager

A Python application with a Streamlit interface for managing slotting operations.

## Project Structure

```
r2bit_Slotting/
├── src/
│   ├── __init__.py
│   └── slotting.py      # Core slotting functionality
├── streamlit_app.py     # Streamlit web interface
├── requirements.txt     # Project dependencies
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd r2bit_Slotting
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file and add your OpenAI API key
# Replace 'your_openai_api_key_here' with your actual API key
```

## Usage

Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

This will start the web interface, which you can access in your browser.

### LLM Analysis

To analyze slotting results using OpenAI's GPT-4o:

```bash
python src/llm_model_analysis.py path/to/slotting_results.csv --abc-file path/to/abc_results.csv
```

Or you can provide the API key directly:

```bash
python src/llm_model_analysis.py path/to/slotting_results.csv --api-key your_openai_api_key
```

## Features

- ABC classification of SKUs based on order frequency and unit volume
- Warehouse slotting with configurable zones (A, B, C)
- Interactive Streamlit interface for parameter configuration
- Visualization of results with charts and tables
- AI-powered analysis of slotting results using OpenAI GPT-4o

## Development

To extend the functionality:
1. Modify the `SlottingManager` class in `src/slotting.py` to add new features
2. Update the Streamlit interface in `streamlit_app.py` to expose the new features
