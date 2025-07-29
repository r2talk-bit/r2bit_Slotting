"""
LLM Model Analysis Module

This module provides functionality to analyze slotting output using OpenAI's GPT-4o model
and generate explanatory summaries of the slotting results.
"""

# ===== BEGINNER'S GUIDE TO AI ANALYSIS IN WAREHOUSE SLOTTING =====
# This module uses Artificial Intelligence (AI) to analyze warehouse slotting results.
# 
# What is AI Analysis?
# - After we've classified items (SKUs) and assigned them to warehouse locations,
#   we want to understand if our slotting plan is good or could be improved.
# - Instead of manually analyzing the data, we use OpenAI's GPT-4o model (an advanced AI)
#   to automatically generate insights and recommendations.
#
# The AI helps answer questions like:
# - How efficient is our proposed warehouse layout?
# - Are the right items in the right zones?
# - What improvements could we make to the slotting plan?
# - Is it worth moving items to new locations?
#
# This module connects to OpenAI's API, sends our slotting data,
# and returns a detailed analysis in Portuguese.
# =================================================================

# Operating system functions (for file paths, environment variables)
import os
# Data analysis library for working with tabular data
import pandas as pd
# Library for converting Python objects to JSON format (for API requests)
import json
# OpenAI's official Python client library
from openai import OpenAI
# Library for parsing command-line arguments
import argparse
# Type hints for better code documentation
from typing import Dict, Any, Optional
# Library for loading environment variables from a .env file
from dotenv import load_dotenv

# Load environment variables from .env file
# This is where we'll get the OpenAI API key from
load_dotenv()


class SlottingAnalyzer:
    """
    Class for analyzing slotting results using OpenAI's GPT-4o model.
    """
    # This is the main class that handles all the AI analysis functionality.
    # It reads slotting data, prepares it for the AI, sends it to OpenAI,
    # and returns the analysis results.
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the SlottingAnalyzer.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will try to get from environment variable.
        """
        # === BEGINNER'S NOTE ===
        # This method sets up the SlottingAnalyzer with an OpenAI API key.
        # The API key is like a password that lets us use OpenAI's AI models.
        # 
        # We can get the API key in two ways:
        # 1. From a .env file (a special file that stores secret information)
        # 2. Passed directly as a parameter when creating the SlottingAnalyzer
        #
        # We prioritize the key from the .env file over the parameter.
        # =====================
        
        print("valor de api_key ", api_key)

        # Force reload from .env file to ensure we have the latest value
        # This helps fix issues where the .env file might have been updated
        load_dotenv(override=True)
        
        # Prioritize the environment variable from .env over the parameter
        # This is the fix for the issue mentioned in the project memory
        self.api_key = os.environ.get("OPENAI_API_KEY") or api_key
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Print the raw API key with visible whitespace markers for debugging
        # This helps identify if there are any hidden spaces or characters in the key
        print(f"Raw API key length: {len(self.api_key)}")
        print(f"First 10 chars (with visible spaces): '{self.api_key[:10].replace(' ', '␣')}'") 
        
        # More aggressive cleaning to handle any whitespace or special characters
        # This removes any extra spaces, quotes, or newlines that might cause API errors
        self.api_key = self.api_key.strip()
        self.api_key = self.api_key.replace('"', '').replace("'", '').replace('\n', '')
        self.api_key = self.api_key.replace(' ', '')  # Remove any spaces within the key
        
        print(f"Cleaned API key length: {len(self.api_key)}")
        print(f"Raw API key length: {len(self.api_key)}")
        print(f"First 10 chars: '{self.api_key[:10]}'")
        
        # Create OpenAI client with the cleaned API key
        # This client will be used to send requests to OpenAI's API
        self.client = OpenAI(api_key=self.api_key)

        print("valor atualizado de api_key", self.api_key)

    def analyze_slotting_output(self, slotting_file: str, abc_file: Optional[str] = None) -> str:
        """
        Analyze slotting output file and generate a summary explanation.
        
        Args:
            slotting_file (str): Path to the slotting output CSV file
            abc_file (str, optional): Path to the ABC classification output CSV file
            
        Returns:
            str: Summary explanation of the slotting results
        """
        # === BEGINNER'S NOTE ===
        # This is the main method that performs the AI analysis. It works in these steps:
        # 1. Read the slotting data from CSV files
        # 2. Calculate statistics about the slotting plan
        # 3. Format the data for the AI to understand
        # 4. Send the data to OpenAI's GPT-4o model
        # 5. Return the AI's analysis
        # =====================
        
        # Read slotting output file
        try:
            # Load the CSV file containing our slotting assignments
            slotting_df = pd.read_csv(slotting_file)
            # Print debug information to help troubleshoot any issues
            print("Available columns in slotting_df:")
            print(slotting_df.columns.tolist())
            print("\nSample data (first 2 rows):")
            print(slotting_df.head(2))
        except Exception as e:
            # If there's an error reading the file, raise a clear error message
            raise ValueError(f"Error reading slotting file: {str(e)}")
        
        # Read ABC classification file if provided
        # This file contains information about which items are A, B, or C class
        abc_df = None
        if abc_file:
            try:
                abc_df = pd.read_csv(abc_file)
            except Exception as e:
                raise ValueError(f"Error reading ABC classification file: {str(e)}")
        
        # Calculate statistics about our slotting plan
        # These statistics help the AI understand the data better
        stats = self._generate_statistics(slotting_df, abc_df)
        
        # Prepare data structures to organize the slotting information
        # We'll separate the original locations from the proposed new locations
        original_slotting_data = {}  # Where items were originally located
        proposed_slotting_data = {}  # Where items are proposed to be located
        movement_cost_details = {}   # Details about moving items from original to proposed locations
        
        # Assume we have data about where items were located before the new slotting plan
        has_pre_slotting_data = True
        
        # Define the potential benefits of the slotting proposal
        # These are written in Portuguese since the analysis will be in Portuguese
        potential_benefits = {
            "Eficiência de Picking": "Melhoria esperada pela alocação de SKUs de alta rotatividade em zonas de fácil acesso",
            "Utilização de Espaço": "Otimização do uso de espaço de acordo com a classificação ABC",
            "Tempo de Ciclo": "Redução esperada no tempo de ciclo de picking",
            "Ergonomia": "Melhoria na ergonomia pela alocação adequada de SKUs por peso e volume"
        }
        
        # Loop through each SKU in the slotting data
        # For each SKU, we extract its original and proposed locations
        for _, row in slotting_df.iterrows():
            sku = row['sku']  # The unique identifier for each item
            
            # Get the proposed new location information for this SKU
            # The .get() method safely returns 'N/A' if the column doesn't exist
            proposed_location = row.get('location', 'N/A')  # Where the item should be placed
            proposed_zone = row.get('zone', 'N/A')          # Which zone (A, B, or C) it belongs in
            abc_class = row.get('abc_class', 'N/A')         # The item's ABC classification
            
            # Get the original location information for this SKU
            # This is where the item was located before the new slotting plan
            original_location = row.get('pre_slotting_location', 'N/A')
            original_zone = row.get('pre_slotting_zone', 'N/A')
            
            # Store the original location data in our dictionary
            # We use the SKU as the key so we can easily look it up later
            original_slotting_data[sku] = {
                'location': original_location,  # The specific storage location
                'zone': original_zone,          # Which zone it was in
                'abc_class': abc_class          # Its ABC classification
            }
            
            # Store the proposed location data in our dictionary
            proposed_slotting_data[sku] = {
                'location': proposed_location,
                'zone': proposed_zone,
                'abc_class': abc_class
            }
            
            # If the item is being moved to a new location, track this movement
            # This helps us analyze the cost of implementing the slotting plan
            if original_location != 'N/A' and proposed_location != 'N/A' and original_location != proposed_location:
                # Record details about this move
                movement_cost_details[sku] = {
                    'from': original_location,    # Where it's moving from
                    'to': proposed_location,      # Where it's moving to
                    'moved': True                 # Flag indicating it needs to be moved
                }

        # Print the data for debugging purposes
        print(original_slotting_data)
        print(proposed_slotting_data)
        
        # Define information about the warehouse layout
        # This is currently a placeholder - in a real system, this would be actual data
        # about the warehouse layout, distances between zones, etc.
        warehouse_layout_data = {
            'layout_description': 'Warehouse layout information would go here',
            # Define the three zones and their characteristics
            'zones': {
                # Zone A is closest to the dock (distance: 10 units)
                'A': {'description': 'Fast-moving items zone', 'distance_from_dock': 10},
                # Zone B is further away (distance: 20 units)
                'B': {'description': 'Medium-moving items zone', 'distance_from_dock': 20},
                # Zone C is furthest away (distance: 30 units)
                'C': {'description': 'Slow-moving items zone', 'distance_from_dock': 30}
            },
            'note': 'This is a placeholder. Replace with actual warehouse layout data when available.'
        }

        # === BEGINNER'S NOTE ===
        # Now we prepare the instructions for the AI (called a "prompt")
        # The system prompt tells the AI what role to play and what to do
        # This prompt is in Portuguese because we want the analysis in Portuguese
        # =====================
        
        # Create prompt for the LLM (Language Learning Model, the AI)
        # This is the "system prompt" that defines the AI's role and task
        system_prompt = """
            Você é um especialista em logística e otimização de armazéns, com profundo conhecimento em análise de dados de slotting de SKUs.
            Sua tarefa é realizar uma análise de impacto detalhada de uma proposta de realocação de SKUs (slotting), comparando o layout original com o proposto.

            O objetivo principal é avaliar o impacto da proposta na **eficiência de picking**, medido primariamente pela **redução da distância total percorrida** pelos operadores.

            Você deve seguir uma abordagem de **Otimização de Pareto**, buscando o melhor equilíbrio entre o **custo de implementação** (medido pelo número de movimentações) e o **ganho de eficiência operacional** (medido pela redução na distância de picking).

            Sua análise deve ser profissional, orientada por dados e acionável. Foque em:

            1.  **Análise de Movimentações:** Quantifique o número de SKUs que permanecem no mesmo local, mudam de endereço dentro da mesma zona, e mudam de zona. Avalie o esforço de implementação com base nisso.
            2.  **Impacto na Eficiência de Picking:** Calcule e compare a distância média de picking para um pedido típico (ou a distância total para um conjunto de pedidos) nos cenários original e proposto. Utilize os dados de coordenadas dos endereços e a frequência de picking dos SKUs para esta análise.
            3.  **Avaliação Custo-Benefício:** Relacione diretamente o número de movimentações necessárias com a redução percentual na distância de picking. Destaque se o esforço de mudança justifica o ganho de eficiência.
            4.  **Análise de Pareto e Cenários Alternativos:** Com base na sua análise, identifique subgrupos de movimentações. Proponha cenários alternativos que possam entregar uma parte significativa do benefício (ex: 80% do ganho de distância) com um esforço muito menor (ex: 50% das movimentações). Por exemplo, sugira focar apenas nas movimentações de SKUs da Curva A ou apenas nas que resultam na maior redução de distância individual.
            5.  **Recomendações e Resumo Executivo:** Conclua com um resumo claro, suas principais descobertas e uma recomendação final sobre a viabilidade da proposta, incluindo sugestões de implementação faseada ou ajustes.

            A análise deve ser escrita em português, de forma clara e direta. Se dados cruciais (como coordenadas) não forem fornecidos, afirme claramente como isso limita sua análise e quais seriam os cálculos possíveis se os tivesse.
            """
        # This is the "user prompt" that contains the specific data and request
        # It includes our statistics, original and proposed slotting data, and potential benefits
        # The f""" syntax allows us to insert variables inside the string using {variable_name}
        user_prompt = f"""
            Realize uma análise de impacto e otimização de Pareto para a proposta de slotting a seguir.

            O objetivo é maximizar a eficiência de picking (redução de distância percorrida) enquanto se considera o esforço de implementação (número de movimentações).

            **Dados para Análise:**

            Slotting Statistics:
            {json.dumps(stats, indent=2)}

            Original Slotting (SKU: Endereço Original, Zona ABC Original):
            {json.dumps(original_slotting_data, indent=2)}

            Proposed Slotting (SKU: Endereço Proposto, Zona ABC Proposta):
            {json.dumps(proposed_slotting_data, indent=2)}

            Benefícios Potenciais da Proposta:
            {json.dumps(potential_benefits, indent=2)}

            **Sua Tarefa:**

            Com base nos dados fornecidos, gere um relatório conciso contendo:

            1.  **Resumo Executivo:** Uma conclusão clara sobre a viabilidade e o impacto da proposta.
            2.  **Análise de Custo de Implementação:** Detalhe o número e o tipo de movimentações necessárias.
            3.  **Análise de Impacto na Eficiência:**
                - Calcule a distância média ponderada de picking para o cenário original.
                - Calcule a distância média ponderada de picking para o cenário proposto.
                - Apresente o ganho percentual de eficiência (redução de distância).
            4.  **Otimização de Pareto e Cenários Alternativos:**
                - Identifique os SKUs cuja movimentação oferece o maior retorno (maior redução de distância por uma única movimentação).
                - Proponha pelo menos um cenário alternativo (ex: "Fase 1") que envolva menos movimentações, mas que capture a maior parte do ganho de eficiência. Justifique sua proposta com dados.

            Seja pragmático e focado em fornecer recomendações acionáveis para a equipe de logística.        
            """

        # This is an alternative user prompt that includes more detailed warehouse layout data
        # It's structured differently to provide more context about the warehouse layout
        # Currently this prompt is defined but not used - it's here for future enhancement
        next_user_prompt = f"""
            Realize uma análise de impacto e otimização de Pareto para a proposta de slotting a seguir.

            O objetivo é maximizar a eficiência de picking (redução de distância percorrida) enquanto se considera o esforço de implementação (número de movimentações).

            **Dados para Análise:**

            1.  **Estatísticas Gerais da Proposta:**
                {json.dumps(stats, indent=2)}

            2.  **Mapeamento do Armazém (Layout e Distâncias):**
                // Esta é a informação complementar mais crítica.
                {json.dumps(warehouse_layout_data, indent=2)}

            3.  **Dados de Slotting Original (SKU, Endereço, Zona, Frequência):**
                // Adicionada a frequência de picking por SKU.
                {json.dumps(original_slotting_data, indent=2)}

            4.  **Dados de Slotting Proposto (SKU, Endereço, Zona, Frequência):**
                {json.dumps(proposed_slotting_data, indent=2)}

            **Sua Tarefa:**

            Com base nos dados fornecidos, gere um relatório conciso contendo:

            1.  **Resumo Executivo:** Uma conclusão clara sobre a viabilidade e o impacto da proposta.
            2.  **Análise de Custo de Implementação:** Detalhe o número e o tipo de movimentações necessárias.
            3.  **Análise de Impacto na Eficiência:**
                - Calcule a distância média ponderada de picking para o cenário original.
                - Calcule a distância média ponderada de picking para o cenário proposto.
                - Apresente o ganho percentual de eficiência (redução de distância).
            4.  **Otimização de Pareto e Cenários Alternativos:**
                - Identifique os SKUs cuja movimentação oferece o maior retorno (maior redução de distância por uma única movimentação).
                - Proponha pelo menos um cenário alternativo (ex: "Fase 1") que envolva menos movimentações, mas que capture a maior parte do ganho de eficiência. Justifique sua proposta com dados.

            Seja pragmático e focado em fornecer recomendações acionáveis para a equipe de logística.
            """

        # === BEGINNER'S NOTE ===
        # Now we send our prompts to OpenAI's API and get back the analysis
        # - model="gpt-4o" specifies which AI model to use (GPT-4o is very advanced)
        # - messages contains our system and user prompts
        # - temperature=0.2 makes the response more focused and less creative
        # - max_tokens=1000 limits the length of the response
        # =====================
        
        # Call OpenAI API to get the AI analysis
        response = self.client.chat.completions.create(
            model="gpt-4o",  # We can also use "gpt-3.5-turbo" which may have wider access
            messages=[
                {"role": "system", "content": system_prompt},  # Instructions for the AI
                {"role": "user", "content": user_prompt}      # Data for the AI to analyze
            ],
            temperature=0.2,    # Lower temperature = more focused, less random
            max_tokens=1000     # Maximum length of the response
        )
        
        # Return just the text content of the AI's response
        # The response object contains metadata, but we only want the actual analysis text
        return response.choices[0].message.content
    
    def _generate_statistics(self, slotting_df: pd.DataFrame, abc_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate statistics from the slotting data for LLM analysis.
        
        Args:
            slotting_df (pd.DataFrame): Slotting results dataframe
            abc_df (pd.DataFrame, optional): ABC classification dataframe
            
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        # === BEGINNER'S NOTE ===
        # This method calculates various statistics about our slotting plan.
        # These statistics help the AI understand the data and provide better analysis.
        # We calculate things like:
        # - How many SKUs (items) are in each zone
        # - How many SKUs are in each ABC class
        # - How ABC classes are distributed across zones
        # =====================
        # Create an empty dictionary to store all our statistics
        stats = {}
        
        # Calculate the total number of SKUs (items) in our dataset
        stats["total_skus"] = len(slotting_df)
        
        # Count how many SKUs are assigned to each zone (A, B, C)
        if "zone" in slotting_df.columns:
            # value_counts() counts how many times each zone appears
            # to_dict() converts the result to a dictionary like {'A': 50, 'B': 30, 'C': 20}
            zone_counts = slotting_df["zone"].value_counts().to_dict()
            stats["zone_distribution"] = zone_counts
            
            # Calculate what percentage of SKUs are in each zone
            # For example: {'A': 50%, 'B': 30%, 'C': 20%}
            stats["zone_utilization"] = {
                zone: count / stats["total_skus"] * 100 
                for zone, count in zone_counts.items()
            }
        
        # Count how many SKUs are in each ABC class (A, B, C)
        if "abc_class" in slotting_df.columns:
            class_counts = slotting_df["abc_class"].value_counts().to_dict()
            stats["abc_class_distribution"] = class_counts
        
        # Create a cross-tabulation to see how ABC classes are distributed across zones
        # This shows, for example, how many Class A items are in Zone A, Zone B, etc.
        # This is important to check if high-importance items (Class A) are in
        # easy-to-access zones (Zone A)
        if "abc_class" in slotting_df.columns and "zone" in slotting_df.columns:
            # crosstab creates a table counting combinations of abc_class and zone
            cross_tab = pd.crosstab(slotting_df["abc_class"], slotting_df["zone"]).to_dict()
            stats["class_zone_distribution"] = {
                str(abc_class): zone_dict 
                for abc_class, zone_dict in cross_tab.items()
            }
        
        # Calculate statistics about the scores (if available)
        # The score represents the importance of each SKU based on order frequency and unit volume
        if "score" in slotting_df.columns:
            stats["score_stats"] = {
                "mean": slotting_df["score"].mean(),      # Average score
                "median": slotting_df["score"].median(),  # Middle value when sorted
                "min": slotting_df["score"].min(),       # Lowest score
                "max": slotting_df["score"].max()        # Highest score
            }
        
        # Calculate statistics about the warehouse locations
        # This counts how many unique values exist for each location component
        # For example, how many different rows ("ruas") are in the warehouse
        if all(col in slotting_df.columns for col in ["deposito", "rua", "bloco", "nivel", "apartamento"]):
            stats["location_stats"] = {
                "unique_depositos": slotting_df["deposito"].nunique(),       # Number of warehouses
                "unique_ruas": slotting_df["rua"].nunique(),                 # Number of rows
                "unique_blocos": slotting_df["bloco"].nunique(),             # Number of blocks
                "unique_niveis": slotting_df["nivel"].nunique(),             # Number of levels
                "unique_apartamentos": slotting_df["apartamento"].nunique(), # Number of apartments
            }
        
        # Return all the calculated statistics
        return stats


def analyze_slotting(slotting_file: str, abc_file: Optional[str] = None, api_key: Optional[str] = None) -> str:
    """
    Convenience function to analyze slotting output and generate a summary.
    
    Args:
        slotting_file (str): Path to the slotting output CSV file
        abc_file (str, optional): Path to the ABC classification output CSV file
        api_key (str, optional): OpenAI API key
        
    Returns:
        str: Summary explanation of the slotting results
    """
    # === BEGINNER'S NOTE ===
    # This is a helper function that makes it easier to use the SlottingAnalyzer.
    # Instead of having to create a SlottingAnalyzer object and then call methods on it,
    # you can just call this function directly with all the parameters.
    # 
    # This is the function that the Streamlit app uses to perform AI analysis.
    # =====================
    
    # Create a SlottingAnalyzer object with the provided API key
    analyzer = SlottingAnalyzer(api_key)
    # Use the analyzer to analyze the slotting output
    return analyzer.analyze_slotting_output(slotting_file, abc_file)


if __name__ == "__main__":
    # === BEGINNER'S NOTE ===
    # This section runs only when you execute this file directly (not when imported).
    # It provides a command-line interface to run AI analysis from the terminal.
    # 
    # Example command:
    # python llm_model_analysis.py slotting_results.csv --abc-file abc_results.csv
    # =====================
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Analyze slotting output using OpenAI GPT-4o')
    parser.add_argument('slotting_file', help='Path to slotting output CSV file')
    parser.add_argument('--abc-file', help='Path to ABC classification output CSV file')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY environment variable)')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    try:
        # Run the analysis with the provided arguments
        summary = analyze_slotting(args.slotting_file, args.abc_file, args.api_key)
        # Print the analysis results
        print("\n=== Slotting Analysis Summary ===\n")
        print(summary)
    except Exception as e:
        # If there's an error, print a helpful error message
        print(f"Error analyzing slotting: {str(e)}")
