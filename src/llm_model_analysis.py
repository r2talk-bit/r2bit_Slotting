"""
LLM Model Analysis Module

This module provides functionality to analyze slotting output using OpenAI's GPT-4o model
and generate explanatory summaries of the slotting results.
"""

import os
import pandas as pd
import json
from openai import OpenAI
import argparse
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class SlottingAnalyzer:
    """
    Class for analyzing slotting results using OpenAI's GPT-4o model.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the SlottingAnalyzer.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will try to get from environment variable.
        """
        print("valor de api_key ", api_key)

        # Force reload from .env file to ensure we have the latest value
        load_dotenv(override=True)
        
        # Prioritize the environment variable from .env over the parameter
        self.api_key = os.environ.get("OPENAI_API_KEY") or api_key
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Print the raw API key with visible whitespace markers for debugging
        print(f"Raw API key length: {len(self.api_key)}")
        print(f"First 10 chars (with visible spaces): '{self.api_key[:10].replace(' ', '␣')}'") 
        
        # More aggressive cleaning to handle any whitespace or special characters
        self.api_key = self.api_key.strip()
        self.api_key = self.api_key.replace('"', '').replace("'", '').replace('\n', '')
        self.api_key = self.api_key.replace(' ', '')  # Remove any spaces within the key
        
        print(f"Cleaned API key length: {len(self.api_key)}")
        print(f"Raw API key length: {len(self.api_key)}")
        print(f"First 10 chars: '{self.api_key[:10]}'")
        
        # Create OpenAI client with the cleaned API key
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
        # Read slotting output file
        try:
            slotting_df = pd.read_csv(slotting_file)
            # Debug: Print column names to check if pre_slotting_location exists
            print("Available columns in slotting_df:")
            print(slotting_df.columns.tolist())
            print("\nSample data (first 2 rows):")
            print(slotting_df.head(2))
        except Exception as e:
            raise ValueError(f"Error reading slotting file: {str(e)}")
        
        # Read ABC classification file if provided
        abc_df = None
        if abc_file:
            try:
                abc_df = pd.read_csv(abc_file)
            except Exception as e:
                raise ValueError(f"Error reading ABC classification file: {str(e)}")
        
        # Generate statistics and insights from the data
        stats = self._generate_statistics(slotting_df, abc_df)
        
        # Extract slotting data for the prompt, separating original and proposed locations
        original_slotting_data = {}
        proposed_slotting_data = {}
        movement_cost_details = {}
        
        # We assume pre_slotting_data will always be available
        has_pre_slotting_data = True
        
        # Define potential benefits
        potential_benefits = {
            "Eficiência de Picking": "Melhoria esperada pela alocação de SKUs de alta rotatividade em zonas de fácil acesso",
            "Utilização de Espaço": "Otimização do uso de espaço de acordo com a classificação ABC",
            "Tempo de Ciclo": "Redução esperada no tempo de ciclo de picking",
            "Ergonomia": "Melhoria na ergonomia pela alocação adequada de SKUs por peso e volume"
        }
        
        for _, row in slotting_df.iterrows():
            sku = row['sku']
            
            # Proposed new location data
            proposed_location = row.get('location', 'N/A')
            proposed_zone = row.get('zone', 'N/A')
            abc_class = row.get('abc_class', 'N/A')
            
            # Original location data
            original_location = row.get('pre_slotting_location', 'N/A')
            original_zone = row.get('pre_slotting_zone', 'N/A')
            
            # Store original data
            original_slotting_data[sku] = {
                'location': original_location,
                'zone': original_zone,
                'abc_class': abc_class
            }
            
            # Store proposed data
            proposed_slotting_data[sku] = {
                'location': proposed_location,
                'zone': proposed_zone,
                'abc_class': abc_class
            }
            
            # Track movements if locations are different
            if original_location != 'N/A' and proposed_location != 'N/A' and original_location != proposed_location:
                # Simple tracking of movements - all movements have the same cost
                movement_cost_details[sku] = {
                    'from': original_location,
                    'to': proposed_location,
                    'moved': True
                }

        print(original_slotting_data)
        print(proposed_slotting_data)
        
        # Define warehouse layout data (placeholder for now)
        warehouse_layout_data = {
            'layout_description': 'Warehouse layout information would go here',
            'zones': {
                'A': {'description': 'Fast-moving items zone', 'distance_from_dock': 10},
                'B': {'description': 'Medium-moving items zone', 'distance_from_dock': 20},
                'C': {'description': 'Slow-moving items zone', 'distance_from_dock': 30}
            },
            'note': 'This is a placeholder. Replace with actual warehouse layout data when available.'
        }

        # Create prompt for the LLM
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

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-4o",  # Changed from gpt-4o to gpt-3.5-turbo which may have wider access
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
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
        stats = {}
        
        # Total SKUs
        stats["total_skus"] = len(slotting_df)
        
        # Zone distribution
        if "zone" in slotting_df.columns:
            zone_counts = slotting_df["zone"].value_counts().to_dict()
            stats["zone_distribution"] = zone_counts
            
            # Zone utilization percentage
            stats["zone_utilization"] = {
                zone: count / stats["total_skus"] * 100 
                for zone, count in zone_counts.items()
            }
        
        # ABC class distribution
        if "abc_class" in slotting_df.columns:
            class_counts = slotting_df["abc_class"].value_counts().to_dict()
            stats["abc_class_distribution"] = class_counts
        
        # Cross-tabulation of ABC class vs Zone
        if "abc_class" in slotting_df.columns and "zone" in slotting_df.columns:
            cross_tab = pd.crosstab(slotting_df["abc_class"], slotting_df["zone"]).to_dict()
            stats["class_zone_distribution"] = {
                str(abc_class): zone_dict 
                for abc_class, zone_dict in cross_tab.items()
            }
        
        # Score statistics if available
        if "score" in slotting_df.columns:
            stats["score_stats"] = {
                "mean": slotting_df["score"].mean(),
                "median": slotting_df["score"].median(),
                "min": slotting_df["score"].min(),
                "max": slotting_df["score"].max()
            }
        
        # Location statistics
        if all(col in slotting_df.columns for col in ["deposito", "rua", "bloco", "nivel", "apartamento"]):
            stats["location_stats"] = {
                "unique_depositos": slotting_df["deposito"].nunique(),
                "unique_ruas": slotting_df["rua"].nunique(),
                "unique_blocos": slotting_df["bloco"].nunique(),
                "unique_niveis": slotting_df["nivel"].nunique(),
                "unique_apartamentos": slotting_df["apartamento"].nunique(),
            }
        
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
    analyzer = SlottingAnalyzer(api_key)
    return analyzer.analyze_slotting_output(slotting_file, abc_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze slotting output using OpenAI GPT-4o')
    parser.add_argument('slotting_file', help='Path to slotting output CSV file')
    parser.add_argument('--abc-file', help='Path to ABC classification output CSV file')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY environment variable)')
    
    args = parser.parse_args()
    
    try:
        summary = analyze_slotting(args.slotting_file, args.abc_file, args.api_key)
        print("\n=== Slotting Analysis Summary ===\n")
        print(summary)
    except Exception as e:
        print(f"Error analyzing slotting: {str(e)}")
