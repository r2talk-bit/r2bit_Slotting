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

## Usage

Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

This will start the web interface, which you can access in your browser.

## Features

- Add new slots with custom properties
- View all existing slots
- Update slot information
- Delete slots when no longer needed

## Development

To extend the functionality:
1. Modify the `SlottingManager` class in `src/slotting.py` to add new features
2. Update the Streamlit interface in `streamlit_app.py` to expose the new features
