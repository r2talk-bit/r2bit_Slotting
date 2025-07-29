"""
Slotting module for the R2Bit Slotting project.
This module contains the core functionality for slotting operations based on ABC classification.
"""

# === BEGINNER'S GUIDE TO WAREHOUSE SLOTTING ===
# Warehouse slotting is the process of deciding where to place products in a warehouse.
# Good slotting can dramatically improve efficiency by:
#   1. Placing fast-moving items in easy-to-reach locations
#   2. Grouping related items together
#   3. Minimizing travel time for warehouse workers
#
# This module implements a slotting strategy based on ABC classification:
#   - A-class items (high importance): placed in prime locations (Zone A)
#   - B-class items (medium importance): placed in secondary locations (Zone B)
#   - C-class items (low importance): placed in remaining locations (Zone C)
#
# The warehouse is modeled as a grid with:
#   - Deposits (deposito): Different warehouse buildings
#   - Rows (rua): Aisles in the warehouse
#   - Blocks (bloco): Sections within a row
#   - Levels (nivel): Vertical position (shelves)
#   - Apartments (apartamento): Positions within a level
# ==========================================

# === BEGINNER'S NOTE: IMPORTS ===
# These are the external libraries and modules that our code needs to function:
#
# pandas (imported as pd): A powerful data analysis library that lets us work with
#                          tabular data (like Excel spreadsheets) in Python.
#                          We use it to read/write CSV files and manipulate SKU data.
#
# os: Provides functions for interacting with the operating system,
#     like working with file paths and checking if files exist.
#
# argparse: Helps create user-friendly command-line interfaces,
#           making it easy for users to run our script with different options.
#
# pathlib.Path: A modern way to handle file paths that works across different
#               operating systems (Windows, Mac, Linux).
# ============================

import pandas as pd  # For data manipulation and analysis
import os            # For file and directory operations
import argparse      # For command-line argument parsing
from pathlib import Path  # For file path handling


class WarehouseLocation:
    """
    Class representing a warehouse location with format: deposito.rua.bloco.nivel.apartamento
    """
    # === BEGINNER'S NOTE ===
    # This class represents a single location in the warehouse where a product (SKU) can be stored.
    # Think of it as a specific address for a product, like:
    # - Deposit 01 (which warehouse building)
    # - Row 02 (which aisle)
    # - Block 03 (which section of the aisle)
    # - Level 01 (how high up on the shelf)
    # - Apartment 04 (which position on that level)
    # 
    # This would be written as: 01.02.03.01.04
    # =====================
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
        # Store all the components of the location address
        self.deposito = deposito      # Which warehouse building (usually 01)
        self.rua = rua                # Which row/aisle in the warehouse
        self.bloco = bloco            # Which block/section in the row
        self.nivel = nivel            # Which vertical level (1 is ground level, higher numbers are higher shelves)
        self.apartamento = apartamento # Which position on the level
        self.sku = None  # No product (SKU) assigned to this location yet
        
    def get_location_id(self):
        """
        Get the location ID in the format 'DD.RR.BB.NN.AA'
        
        Returns:
            str: Formatted location ID
        """
        # Format each component as a 2-digit number with leading zeros
        # For example, deposit 1 becomes '01', row 2 becomes '02', etc.
        # The f-string with :02d means "format as 2-digit integer with leading zeros"
        return f"{self.deposito:02d}.{self.rua:02d}.{self.bloco:02d}.{self.nivel:02d}.{self.apartamento:02d}"
    
    def is_available(self):
        """Check if the location is available (no SKU assigned)"""
        # A location is available if no SKU (product) has been assigned to it
        return self.sku is None
    
    def assign_sku(self, sku):
        """
        Assign an SKU to this location
        
        Args:
            sku (str): SKU identifier
            
        Returns:
            bool: True if successful, False if location is already occupied
        """
        # First check if the location is already occupied by another product
        if not self.is_available():
            # If occupied, we can't assign a new SKU here
            return False
        
        # Assign the SKU to this location
        self.sku = sku
        return True
    
    def __str__(self):
        # This special method controls what happens when you print this object
        # or convert it to a string with str()
        return self.get_location_id()
    
    def __repr__(self):
        # This special method controls how this object is represented
        # in the Python interpreter and in error messages
        return self.get_location_id()


class SlottingManager:
    """
    Main class for handling slotting operations.
    """
    # === BEGINNER'S NOTE ===
    # This is the main class that handles the entire slotting process.
    # It creates a virtual model of your warehouse with all possible storage locations,
    # divides these locations into zones (A, B, C) based on accessibility,
    # and then assigns products to locations based on their ABC classification.
    #
    # Zone A: Prime locations (closest to picking areas, easiest to access)
    # Zone B: Secondary locations (medium access difficulty)
    # Zone C: Tertiary locations (furthest, hardest to access)
    #
    # The goal is to place high-importance products (Class A) in Zone A locations,
    # medium-importance products (Class B) in Zone B locations, and
    # low-importance products (Class C) in Zone C locations.
    # =====================
    
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
        # === BEGINNER'S NOTE ===
        # These parameters define the size and layout of your warehouse:
        # - num_ruas: How many aisles/rows your warehouse has
        # - num_blocos: How many sections/blocks in each row
        # - num_niveis: How many vertical levels (shelves)
        # - num_apartamentos: How many positions on each level
        #
        # And these parameters define how the zones are created:
        # - zona_a_ruas: How many rows are in the prime Zone A
        # - zona_a_blocos: How many blocks in each row are in Zone A
        # - zona_a_niveis: How many levels are in Zone A (usually lower levels for easy access)
        # - zona_b_percentage: What percentage of the remaining locations become Zone B
        # =====================
        
        self.num_ruas = num_ruas                  # Total number of rows in the warehouse
        self.num_blocos = num_blocos              # Total number of blocks per row
        self.num_niveis = num_niveis              # Total number of levels per block
        self.num_apartamentos = num_apartamentos  # Total number of apartments per level
        self.zona_a_ruas = zona_a_ruas            # Number of rows in Zone A
        self.zona_a_blocos = zona_a_blocos        # Number of blocks in Zone A
        self.zona_a_niveis = zona_a_niveis        # Number of levels in Zone A
        self.zona_b_percentage = zona_b_percentage # Percentage for Zone B
        
        # Generate all possible locations in the warehouse
        # This creates a list of all storage positions
        self.locations = self._generate_locations()
        
        # Divide all locations into three zones (A, B, C)
        # based on their proximity to the picking area
        self.zone_a, self.zone_b, self.zone_c = self._define_zones()
        
        # This dictionary will keep track of which SKU is assigned to which location
        # Format: {"SKU001": WarehouseLocation(1,1,1,1,1), ...}
        self.sku_to_location = {}
        
    def _generate_locations(self):
        """
        Generate all possible locations in the warehouse.
        
        Returns:
            list: List of WarehouseLocation objects
        """
        # === BEGINNER'S NOTE ===
        # This method creates all possible storage locations in the warehouse.
        # It uses nested loops to generate every combination of:
        # - row (rua)
        # - block (bloco)
        # - level (nivel)
        # - apartment (apartamento)
        #
        # For example, with 10 rows, 10 blocks, 5 levels, and 10 apartments,
        # this will create 10 × 10 × 5 × 10 = 5,000 unique locations!
        # =====================
        
        locations = []  # Empty list to store all locations
        deposito = 1    # Always use deposit 01
        
        # Create nested loops to generate all possible combinations
        for rua in range(1, self.num_ruas + 1):           # For each row
            for bloco in range(1, self.num_blocos + 1):   # For each block in that row
                for nivel in range(1, self.num_niveis + 1): # For each level in that block
                    for apartamento in range(1, self.num_apartamentos + 1): # For each apartment on that level
                        # Create a new location with these coordinates
                        location = WarehouseLocation(deposito, rua, bloco, nivel, apartamento)
                        # Add it to our list of locations
                        locations.append(location)
        
        return locations
    
    def _define_zones(self):
        """
        Define zones A, B, and C based on proximity to the start of the warehouse.
        
        Returns:
            tuple: Three lists containing locations for zones A, B, and C
        """
        # === BEGINNER'S NOTE ===
        # This method divides all warehouse locations into three zones:
        #
        # Zone A: Prime locations - closest to picking area, easiest to access
        #         (first few rows, blocks, and lower levels)
        # Zone B: Secondary locations - medium distance and accessibility
        # Zone C: Tertiary locations - furthest away, hardest to access
        #
        # The idea is to place high-frequency items (Class A) in Zone A,
        # medium-frequency items (Class B) in Zone B, and
        # low-frequency items (Class C) in Zone C.
        # =====================
        
        zone_a = []      # Will hold all Zone A locations
        non_zone_a = []  # Will hold all locations not in Zone A
        
        # Define Zone A: first X rows, Y blocks, and Z levels
        # These are the prime locations closest to the picking area
        for location in self.locations:
            # Check if this location should be in Zone A
            if location.rua <= self.zona_a_ruas and location.bloco <= self.zona_a_blocos and location.nivel <= self.zona_a_niveis:
                zone_a.append(location)  # Add to Zone A
            else:
                non_zone_a.append(location)  # Not in Zone A
        
        # Sort non-Zone A locations by proximity (row, block, level, apartment)
        # This ensures that Zone B gets the next best locations after Zone A
        non_zone_a.sort(key=lambda loc: (loc.rua, loc.bloco, loc.nivel, loc.apartamento))
        
        # Split remaining locations between Zone B and Zone C
        # Zone B gets a percentage (e.g., 30%) of the best remaining locations
        zone_b_count = int(len(non_zone_a) * self.zona_b_percentage)
        zone_b = non_zone_a[:zone_b_count]  # First portion goes to Zone B
        zone_c = non_zone_a[zone_b_count:]  # Remainder goes to Zone C
        
        return zone_a, zone_b, zone_c
    
    def get_available_locations(self, zone):
        """
        Get available locations in the specified zone.
        
        Args:
            zone (list): List of locations in the zone
            
        Returns:
            list: List of available locations in the zone
        """
        # === BEGINNER'S NOTE ===
        # This method finds all locations in a zone that don't have a product assigned yet.
        # It uses a list comprehension (a compact way to create a new list).
        # The expression [loc for loc in zone if loc.is_available()] means:
        # "Create a list of all locations in the zone where is_available() returns True"
        # =====================
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
        # === BEGINNER'S NOTE ===
        # This method does two things:
        # 1. Tries to assign the SKU to the location using the location's assign_sku method
        # 2. If successful, records this assignment in the sku_to_location dictionary
        #    so we can easily look up where each SKU is stored
        # =====================
        
        # Try to assign the SKU to the location
        if location.assign_sku(sku):
            # If successful, record this assignment in our dictionary
            self.sku_to_location[sku] = location
            return True
        # If the assignment failed (location already occupied), return False
        return False
    
    def perform_slotting(self, skus_df):
        """
        Perform slotting operation for all SKUs.
        
        Args:
            skus_df (pandas.DataFrame): DataFrame containing SKUs with abc_class and score
            
        Returns:
            pandas.DataFrame: DataFrame with slotting results
        """
        # === BEGINNER'S NOTE ===
        # This is the heart of the slotting algorithm. Here's what it does:
        #
        # 1. Sort products by importance (score and ABC class)
        # 2. For each product, try to assign it to the appropriate zone:
        #    - A-class items: try Zone A first, then B, then C
        #    - B-class items: try Zone B first, then C, then A
        #    - C-class items: try Zone C first, then B, then A
        # 3. Record the assignments and create a report
        #
        # The goal is to ensure the most important products get the best locations,
        # but also to make sure every product gets assigned somewhere if possible.
        # =====================
        
        # First, sort the SKUs by score (highest first) and then by ABC class
        # This ensures that the most important items are processed first
        sorted_skus = skus_df.sort_values(['score', 'abc_class'], 
                                         ascending=[False, True]).reset_index(drop=True)
        
        # Create an empty list to store the results for each SKU
        results = []
        
        # Process each SKU one by one, in order of importance
        for _, row in sorted_skus.iterrows():
            # Get the SKU ID and its ABC classification
            sku = row['sku']            # The unique product identifier
            abc_class = row['abc_class'] # The ABC class (A, B, or C)
            
            # Try to assign the SKU to the appropriate zone based on its ABC class
            # We'll track whether we successfully assigned it somewhere
            assigned = False
            
            # For Class A items (highest importance)
            if abc_class == 'A':
                # === BEGINNER'S NOTE ===
                # Class A items are the most important, so we try to place them in Zone A first.
                # If Zone A is full, we try Zone B, then Zone C as a last resort.
                # This ensures that high-importance items get the best locations possible.
                # =====================
                
                # First choice: Try Zone A (prime locations)
                available_a = self.get_available_locations(self.zone_a)
                if available_a:  # If there are available locations in Zone A
                    location = available_a[0]  # Take the first available location
                    self.assign_sku_to_location(sku, location)  # Assign the SKU to this location
                    assigned = True  # Mark as assigned
                else:  # If Zone A is full
                    # Second choice: Try Zone B
                    available_b = self.get_available_locations(self.zone_b)
                    if available_b:  # If there are available locations in Zone B
                        location = available_b[0]  # Take the first available location
                        self.assign_sku_to_location(sku, location)  # Assign the SKU to this location
                        assigned = True  # Mark as assigned
                    else:  # If Zone B is also full
                        # Last resort: Try Zone C
                        available_c = self.get_available_locations(self.zone_c)
                        if available_c:  # If there are available locations in Zone C
                            location = available_c[0]  # Take the first available location
                            self.assign_sku_to_location(sku, location)  # Assign the SKU to this location
                            assigned = True  # Mark as assigned
            
            # For Class B items (medium importance)
            elif abc_class == 'B':
                # === BEGINNER'S NOTE ===
                # Class B items are medium importance, so we try to place them in Zone B first.
                # If Zone B is full, we try Zone C, then Zone A as a last resort.
                # We prefer not to use Zone A for Class B items to save those prime
                # locations for Class A items.
                # =====================
                
                # First choice: Try Zone B (secondary locations)
                available_b = self.get_available_locations(self.zone_b)
                if available_b:  # If there are available locations in Zone B
                    location = available_b[0]  # Take the first available location
                    self.assign_sku_to_location(sku, location)  # Assign the SKU to this location
                    assigned = True  # Mark as assigned
                else:  # If Zone B is full
                    # Second choice: Try Zone C
                    available_c = self.get_available_locations(self.zone_c)
                    if available_c:  # If there are available locations in Zone C
                        location = available_c[0]  # Take the first available location
                        self.assign_sku_to_location(sku, location)  # Assign the SKU to this location
                        assigned = True  # Mark as assigned
                    else:  # If Zone C is also full
                        # Last resort: Try Zone A
                        available_a = self.get_available_locations(self.zone_a)
                        if available_a:  # If there are available locations in Zone A
                            location = available_a[0]  # Take the first available location
                            self.assign_sku_to_location(sku, location)  # Assign the SKU to this location
                            assigned = True  # Mark as assigned
            
            # For Class C items (lowest importance)
            else:  # Class C
                # === BEGINNER'S NOTE ===
                # Class C items are the least important, so we try to place them in Zone C first.
                # If Zone C is full, we try Zone B, then Zone A as a last resort.
                # We prefer to save Zone A and B for more important items.
                # =====================
                
                # First choice: Try Zone C (tertiary locations)
                available_c = self.get_available_locations(self.zone_c)
                if available_c:  # If there are available locations in Zone C
                    location = available_c[0]  # Take the first available location
                    self.assign_sku_to_location(sku, location)  # Assign the SKU to this location
                    assigned = True  # Mark as assigned
                else:  # If Zone C is full
                    # Second choice: Try Zone B
                    available_b = self.get_available_locations(self.zone_b)
                    if available_b:  # If there are available locations in Zone B
                        location = available_b[0]  # Take the first available location
                        self.assign_sku_to_location(sku, location)  # Assign the SKU to this location
                        assigned = True  # Mark as assigned
                    else:  # If Zone B is also full
                        # Last resort: Try Zone A
                        available_a = self.get_available_locations(self.zone_a)
                        if available_a:  # If there are available locations in Zone A
                            location = available_a[0]  # Take the first available location
                            self.assign_sku_to_location(sku, location)  # Assign the SKU to this location
                            assigned = True  # Mark as assigned
            
            # === BEGINNER'S NOTE ===
            # Now that we've tried to assign the SKU to a location,
            # we need to record the result, whether successful or not.
            # We'll create a dictionary with all the information about this SKU
            # and its assigned location (or lack thereof).
            # =====================
            
            # Record the result of the assignment
            if assigned:  # If we successfully assigned the SKU to a location
                # Get the formatted location ID (e.g., "01.02.03.01.04")
                location_id = self.sku_to_location[sku].get_location_id()
                
                # Determine which zone the location is in
                # This uses a conditional expression (ternary operator)
                zone = 'A' if self.sku_to_location[sku] in self.zone_a else \
                       'B' if self.sku_to_location[sku] in self.zone_b else 'C'
            else:  # If we couldn't assign the SKU anywhere
                location_id = "Not assigned"  # Mark as not assigned
                zone = "None"  # No zone
            
            # Create a dictionary with the SKU information and assignment results
            result = {
                'sku': sku,                # The SKU identifier
                'abc_class': abc_class,     # The ABC classification
                'score': row['score'],      # The importance score
                'location': location_id,    # The assigned location (or "Not assigned")
                'zone': zone                # The zone (A, B, C, or "None")
            }
            
            # Copy any other columns from the original data
            # This preserves information like pre_slotting_location, description, etc.
            for col in row.index:
                # If the column isn't already in our result and isn't one we've already handled
                if col not in result and col not in ['sku', 'abc_class', 'score']:
                    result[col] = row[col]  # Copy the value
            
            # Add this SKU's result to our list of results
            results.append(result)
        
        # Convert our list of results into a pandas DataFrame and return it
        # This DataFrame will contain all SKUs with their assigned locations and zones
        return pd.DataFrame(results)

    def generate_report(self, input_file, output_file):
        """
        Generate a slotting report from an input file and save to an output file.
        
        Args:
            input_file (str): Path to the input CSV file
            output_file (str): Path to save the output CSV file
            
        Returns:
            pandas.DataFrame: DataFrame with slotting results
        """
        # === BEGINNER'S NOTE ===
        # This method handles the entire process of reading an input file,
        # performing the slotting operation, and saving the results to an output file.
        # It's the main entry point for the slotting process when working with files.
        #
        # The method performs these key steps:
        # 1. Detects whether the CSV uses commas or semicolons as separators
        # 2. Reads the input file into a pandas DataFrame
        # 3. Validates that all required columns are present
        # 4. Calls perform_slotting() to assign locations to SKUs
        # 5. Saves the results to the output file
        # =====================
        
        try:
            # Detect whether the CSV uses semicolons or commas as separators
            # This makes the program more flexible with different CSV formats
            with open(input_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()  # Read the header line
                separator = ';' if ';' in first_line else ','  # Determine separator
            
            # Read the input CSV file into a pandas DataFrame
            df = pd.read_csv(input_file, sep=separator)
            
            # Check that all required columns are present in the input file
            # The slotting algorithm needs at minimum: sku, abc_class, and score
            required_columns = ['sku', 'abc_class', 'score']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in input file")
            
            # Perform the actual slotting operation
            # This will assign warehouse locations to each SKU
            result_df = self.perform_slotting(df)
            
            # Save the results to the output CSV file
            # We use the same separator (comma or semicolon) as the input file
            result_df.to_csv(output_file, sep=separator, index=False)
            
            return result_df  # Return the results DataFrame for further processing if needed
        
        except Exception as e:
            # If anything goes wrong, print an error message and re-raise the exception
            print(f"Error generating report: {e}")
            raise  # Re-raise the exception so the caller knows something went wrong


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
    """
    # === BEGINNER'S NOTE ===
    # This is a helper function that makes it easier to use the slotting system.
    # Instead of having to create a SlottingManager object and call its methods,
    # you can just call this function with the input and output file paths
    # and any custom warehouse parameters.
    #
    # This is particularly useful for command-line usage or when integrating
    # with other systems that don't need to interact with the SlottingManager directly.
    # =====================
    
    # Create a SlottingManager with the specified parameters
    # This initializes the warehouse model with the given dimensions and zone definitions
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
    
    # Generate the report by reading the input file, performing slotting, and saving results
    return manager.generate_report(input_file, output_file)


if __name__ == "__main__":
    # === BEGINNER'S NOTE ===
    # This section runs when the script is executed directly from the command line.
    # It sets up a command-line interface so users can run the slotting process
    # without having to write any Python code.
    #
    # For example, you can run:
    # python slotting.py input.csv output.csv --num_ruas 15 --zona_a_ruas 4
    #
    # This would process input.csv, create a warehouse with 15 rows (with 4 rows in Zone A),
    # and save the results to output.csv.
    # =====================
    
    import argparse
    
    # Parse command line arguments
    # This creates a user-friendly interface for running the script from the command line
    parser = argparse.ArgumentParser(description='Perform warehouse slotting based on ABC classification.')
    
    # Required arguments - must be provided when running the script
    parser.add_argument('input_file', help='Path to the input CSV file with ABC classification')
    parser.add_argument('output_file', help='Path to save the output CSV file with slotting results')
    
    # Optional arguments - will use default values if not provided
    parser.add_argument('--num_ruas', type=int, default=10, help='Number of rows in the warehouse')
    parser.add_argument('--num_blocos', type=int, default=10, help='Number of blocks per row')
    parser.add_argument('--num_niveis', type=int, default=5, help='Number of levels per block')
    parser.add_argument('--num_apartamentos', type=int, default=10, help='Number of apartments per level')
    parser.add_argument('--zona_a_ruas', type=int, default=3, help='Number of rows to be considered as Zone A')
    parser.add_argument('--zona_a_blocos', type=int, default=3, help='Number of blocks per row to be considered as Zone A')
    parser.add_argument('--zona_a_niveis', type=int, default=2, help='Number of levels to be considered as Zone A')
    parser.add_argument('--zona_b_percentage', type=float, default=0.3, help='Percentage of non-Zone A locations to be assigned to Zone B')
    
    # Parse the arguments provided by the user
    args = parser.parse_args()
    
    # Perform slotting with the provided or default parameters
    perform_slotting(
        args.input_file,          # Input CSV file with SKU data
        args.output_file,         # Where to save the results
        args.num_ruas,            # Warehouse dimensions
        args.num_blocos,
        args.num_niveis,
        args.num_apartamentos,
        args.zona_a_ruas,         # Zone A definition
        args.zona_a_blocos,
        args.zona_a_niveis,
        args.zona_b_percentage    # Zone B percentage
    )
    
    print(f"Slotting complete. Results saved to {args.output_file}")
