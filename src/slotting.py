"""
Slotting module for the R2Bit Slotting project.
This module contains the core functionality for slotting operations based on ABC classification.
"""

import pandas as pd
import os
import argparse
from pathlib import Path


class WarehouseLocation:
    """
    Class representing a warehouse location with format: deposito.rua.bloco.nivel.apartamento
    """
    def __init__(self, deposito, rua, bloco, nivel, apartamento):
        """
        Initialize a warehouse location.
        
        Args:
            deposito (int): Deposit number
            rua (int): Row number
            bloco (int): Block number
            nivel (int): Level number
            apartamento (int): Apartment number
        """
        self.deposito = deposito
        self.rua = rua
        self.bloco = bloco
        self.nivel = nivel
        self.apartamento = apartamento
        self.sku = None  # SKU assigned to this location
        
    def get_location_id(self):
        """
        Get the location ID in the format 'DD.RR.BB.NN.AA'
        
        Returns:
            str: Formatted location ID
        """
        return f"{self.deposito:02d}.{self.rua:02d}.{self.bloco:02d}.{self.nivel:02d}.{self.apartamento:02d}"
    
    def is_available(self):
        """Check if the location is available (no SKU assigned)"""
        return self.sku is None
    
    def assign_sku(self, sku):
        """
        Assign an SKU to this location
        
        Args:
            sku (str): SKU identifier
            
        Returns:
            bool: True if successful, False if location is already occupied
        """
        if not self.is_available():
            return False
        
        self.sku = sku
        return True
    
    def __str__(self):
        return self.get_location_id()
    
    def __repr__(self):
        return self.get_location_id()


class SlottingManager:
    """
    Main class for handling slotting operations.
    """
    
    def __init__(self, num_ruas=10, num_blocos=10, num_niveis=5, num_apartamentos=10,
                 zona_a_ruas=3, zona_a_blocos=3, zona_a_niveis=2, zona_b_percentage=0.3):
        """
        Initialize the SlottingManager.
        
        Args:
            num_ruas (int): Number of rows in the warehouse
            num_blocos (int): Number of blocks per row
            num_niveis (int): Number of levels per block
            num_apartamentos (int): Number of apartments per level
            zona_a_ruas (int): Number of rows to be considered as Zone A
            zona_a_blocos (int): Number of blocks per row to be considered as Zone A
            zona_a_niveis (int): Number of levels to be considered as Zone A
            zona_b_percentage (float): Percentage of non-Zone A locations to be assigned to Zone B
        """
        self.num_ruas = num_ruas
        self.num_blocos = num_blocos
        self.num_niveis = num_niveis
        self.num_apartamentos = num_apartamentos
        self.zona_a_ruas = zona_a_ruas
        self.zona_a_blocos = zona_a_blocos
        self.zona_a_niveis = zona_a_niveis
        self.zona_b_percentage = zona_b_percentage
        
        # Generate all locations
        self.locations = self._generate_locations()
        
        # Define zones
        self.zone_a, self.zone_b, self.zone_c = self._define_zones()
        
        # Dictionary to store SKU to location mapping
        self.sku_to_location = {}
        
    def _generate_locations(self):
        """
        Generate all possible locations in the warehouse.
        
        Returns:
            list: List of WarehouseLocation objects
        """
        locations = []
        deposito = 1  # Always use deposit 01
        
        for rua in range(1, self.num_ruas + 1):
            for bloco in range(1, self.num_blocos + 1):
                for nivel in range(1, self.num_niveis + 1):
                    for apartamento in range(1, self.num_apartamentos + 1):
                        location = WarehouseLocation(deposito, rua, bloco, nivel, apartamento)
                        locations.append(location)
        
        return locations
    
    def _define_zones(self):
        """
        Define zones A, B, and C based on proximity to the start of the warehouse.
        
        Returns:
            tuple: Three lists containing locations for zones A, B, and C
        """
        zone_a = []
        non_zone_a = []
        
        # Define Zone A: first X rows and Y levels
        for location in self.locations:
            if location.rua <= self.zona_a_ruas and location.bloco <= self.zona_a_blocos and location.nivel <= self.zona_a_niveis:
                zone_a.append(location)
            else:
                non_zone_a.append(location)
        
        # Sort non-Zone A locations by proximity (row, block, level, apartment)
        non_zone_a.sort(key=lambda loc: (loc.rua, loc.bloco, loc.nivel, loc.apartamento))
        
        # Split remaining locations between Zone B and Zone C
        zone_b_count = int(len(non_zone_a) * self.zona_b_percentage)
        zone_b = non_zone_a[:zone_b_count]
        zone_c = non_zone_a[zone_b_count:]
        
        return zone_a, zone_b, zone_c
    
    def get_available_locations(self, zone):
        """
        Get available locations in the specified zone.
        
        Args:
            zone (list): List of locations in the zone
            
        Returns:
            list: List of available locations in the zone
        """
        return [loc for loc in zone if loc.is_available()]
    
    def assign_sku_to_location(self, sku, location):
        """
        Assign an SKU to a location.
        
        Args:
            sku (str): SKU identifier
            location (WarehouseLocation): Location to assign the SKU to
            
        Returns:
            bool: True if successful, False otherwise
        """
        if location.assign_sku(sku):
            self.sku_to_location[sku] = location
            return True
        return False
    
    def perform_slotting(self, skus_df):
        """
        Perform slotting operation for all SKUs.
        
        Args:
            skus_df (pandas.DataFrame): DataFrame containing SKUs with abc_class and score
            
        Returns:
            pandas.DataFrame: DataFrame with slotting results
        """
        # Sort SKUs by score (descending) and then by ABC class
        sorted_skus = skus_df.sort_values(['score', 'abc_class'], 
                                         ascending=[False, True]).reset_index(drop=True)
        
        results = []
        
        # Process each SKU
        for _, row in sorted_skus.iterrows():
            sku = row['sku']
            abc_class = row['abc_class']
            
            # Try to assign to the appropriate zone based on ABC class
            assigned = False
            
            if abc_class == 'A':
                # Try Zone A first, then B, then C
                available_a = self.get_available_locations(self.zone_a)
                if available_a:
                    location = available_a[0]
                    self.assign_sku_to_location(sku, location)
                    assigned = True
                else:
                    available_b = self.get_available_locations(self.zone_b)
                    if available_b:
                        location = available_b[0]
                        self.assign_sku_to_location(sku, location)
                        assigned = True
                    else:
                        available_c = self.get_available_locations(self.zone_c)
                        if available_c:
                            location = available_c[0]
                            self.assign_sku_to_location(sku, location)
                            assigned = True
            
            elif abc_class == 'B':
                # Try Zone B first, then C, then A
                available_b = self.get_available_locations(self.zone_b)
                if available_b:
                    location = available_b[0]
                    self.assign_sku_to_location(sku, location)
                    assigned = True
                else:
                    available_c = self.get_available_locations(self.zone_c)
                    if available_c:
                        location = available_c[0]
                        self.assign_sku_to_location(sku, location)
                        assigned = True
                    else:
                        available_a = self.get_available_locations(self.zone_a)
                        if available_a:
                            location = available_a[0]
                            self.assign_sku_to_location(sku, location)
                            assigned = True
            
            else:  # Class C
                # Try Zone C first, then B, then A
                available_c = self.get_available_locations(self.zone_c)
                if available_c:
                    location = available_c[0]
                    self.assign_sku_to_location(sku, location)
                    assigned = True
                else:
                    available_b = self.get_available_locations(self.zone_b)
                    if available_b:
                        location = available_b[0]
                        self.assign_sku_to_location(sku, location)
                        assigned = True
                    else:
                        available_a = self.get_available_locations(self.zone_a)
                        if available_a:
                            location = available_a[0]
                            self.assign_sku_to_location(sku, location)
                            assigned = True
            
            # Record the result
            if assigned:
                location_id = self.sku_to_location[sku].get_location_id()
                zone = 'A' if self.sku_to_location[sku] in self.zone_a else \
                       'B' if self.sku_to_location[sku] in self.zone_b else 'C'
            else:
                location_id = "Not assigned"
                zone = "None"
            
            result = {
                'sku': sku,
                'abc_class': abc_class,
                'score': row['score'],
                'location': location_id,
                'zone': zone
            }
            
            # Add other columns from the original data
            for col in row.index:
                if col not in result and col not in ['sku', 'abc_class', 'score']:
                    result[col] = row[col]
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def generate_report(self, input_file, output_file):
        """
        Generate a slotting report based on an input file with ABC classification.
        
        Args:
            input_file (str): Path to the input CSV file with ABC classification
            output_file (str): Path to save the output CSV file with slotting results
            
        Returns:
            pandas.DataFrame: DataFrame with slotting results
        """
        try:
            # Try to detect the separator (comma or semicolon)
            with open(input_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if ';' in first_line:
                    separator = ';'
                else:
                    separator = ','
            
            # Read the input file with the detected separator
            df = pd.read_csv(input_file, sep=separator)
            
            # Validate required columns exist
            required_columns = ['sku', 'abc_class', 'score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Perform slotting
            result_df = self.perform_slotting(df)
            
            # Save to output file with the same separator as input
            result_df.to_csv(output_file, index=False, sep=separator)
            
            return result_df
            
        except Exception as e:
            raise Exception(f"Error generating slotting report: {str(e)}")


def perform_slotting(input_file, output_file, num_ruas=10, num_blocos=10, num_niveis=5, 
                    num_apartamentos=10, zona_a_ruas=3, zona_a_blocos=3, zona_a_niveis=2, zona_b_percentage=0.3):
    """
    Convenience function to perform slotting based on an ABC classification file.
    
    Args:
        input_file (str): Path to the input CSV file with ABC classification
        output_file (str): Path to save the output CSV file with slotting results
        num_ruas (int): Number of rows in the warehouse
        num_blocos (int): Number of blocks per row
        num_niveis (int): Number of levels per block
        num_apartamentos (int): Number of apartments per level
        zona_a_ruas (int): Number of rows to be considered as Zone A
        zona_a_blocos (int): Number of blocks per row to be considered as Zone A
        zona_a_niveis (int): Number of levels to be considered as Zone A
        zona_b_percentage (float): Percentage of non-Zone A locations to be assigned to Zone B
        
    Returns:
        pandas.DataFrame: DataFrame with slotting results
    """
    manager = SlottingManager(
        num_ruas=num_ruas,
        num_blocos=num_blocos,
        num_niveis=num_niveis,
        num_apartamentos=num_apartamentos,
        zona_a_ruas=zona_a_ruas,
        zona_a_blocos=zona_a_blocos,
        zona_a_niveis=zona_a_niveis,
        zona_b_percentage=zona_b_percentage
    )
    
    return manager.generate_report(input_file, output_file)


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description='Perform slotting based on ABC classification')
    parser.add_argument('input_file', help='Path to input CSV file with ABC classification')
    parser.add_argument('output_file', help='Path to output CSV file for slotting results')
    parser.add_argument('--num-ruas', type=int, default=10, help='Number of rows in the warehouse')
    parser.add_argument('--num-blocos', type=int, default=10, help='Number of blocks per row')
    parser.add_argument('--num-niveis', type=int, default=5, help='Number of levels per block')
    parser.add_argument('--num-apartamentos', type=int, default=10, help='Number of apartments per level')
    parser.add_argument('--zona-a-ruas', type=int, default=3, help='Number of rows for Zone A')
    parser.add_argument('--zona-a-blocos', type=int, default=3, help='Number of blocks per row for Zone A')
    parser.add_argument('--zona-a-niveis', type=int, default=2, help='Number of levels for Zone A')
    parser.add_argument('--zona-b-percentage', type=float, default=0.3, 
                        help='Percentage of non-Zone A locations for Zone B (0-1)')
    
    args = parser.parse_args()
    
    # Validate percentage
    if not 0 < args.zona_b_percentage < 1:
        parser.error("Zone B percentage must be between 0 and 1")
    
    # Run slotting
    perform_slotting(
        input_file=args.input_file,
        output_file=args.output_file,
        num_ruas=args.num_ruas,
        num_blocos=args.num_blocos,
        num_niveis=args.num_niveis,
        num_apartamentos=args.num_apartamentos,
        zona_a_ruas=args.zona_a_ruas,
        zona_a_blocos=args.zona_a_blocos,
        zona_a_niveis=args.zona_a_niveis,
        zona_b_percentage=args.zona_b_percentage
    )
    
    print(f"Slotting complete. Results saved to {args.output_file}")
