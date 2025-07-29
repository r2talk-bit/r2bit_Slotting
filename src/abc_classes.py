"""
ABC Classification module for the R2Bit Slotting project.
This module provides functionality to classify SKUs using the ABC method
based on order frequency and unit volume.
"""

# ===== BEGINNER'S GUIDE TO ABC CLASSIFICATION =====
# ABC classification is a popular inventory management technique that helps warehouses
# prioritize which items (SKUs) need more attention and better placement.
#
# The basic idea is:
# - Class A: Most important items (typically 20% of items that generate 80% of business)
# - Class B: Moderately important items
# - Class C: Least important items (typically many items that generate little business)
#
# This module helps calculate which items belong in which class based on:
# 1. How often the item is ordered (order frequency)
# 2. How many units of the item are ordered (unit volume)
# ===================================================

# Pandas is used for data manipulation and analysis
import pandas as pd
# NumPy is used for numerical operations
import numpy as np


class ABCClassifier:
    """
    Class for performing ABC classification on SKU data.
    
    ABC classification is a method to categorize inventory items according
    to their importance. Items are typically classified as:
    - A: High importance (top 80% by default)
    - B: Medium importance (next 15% by default)
    - C: Low importance (remaining 5% by default)
    """
    # This class contains all the logic needed to classify SKUs into A, B, and C categories.
    # It's the main component of this module and handles all the calculations.
    
    def __init__(self, order_weight=0.7, unit_weight=0.3, 
                 class_a_cutoff=0.8, class_b_cutoff=0.95):
        """
        Initialize the ABC Classifier.
        
        Args:
            order_weight (float): Weight given to order frequency in the score calculation (0-1)
            unit_weight (float): Weight given to unit volume in the score calculation (0-1)
            class_a_cutoff (float): Cumulative percentage cutoff for Class A (0-1)
            class_b_cutoff (float): Cumulative percentage cutoff for Class B (0-1)
                                    Class C will be everything above this cutoff
        """
        # === BEGINNER'S NOTE ===
        # These parameters control how we classify items:
        # - order_weight: How much importance to give to order frequency (default: 70%)
        # - unit_weight: How much importance to give to unit volume (default: 30%)
        # - class_a_cutoff: What percentage of total value should be Class A (default: 80%)
        # - class_b_cutoff: What percentage of total value should be Class A+B (default: 95%)
        # =====================
        
        self.order_weight = order_weight
        self.unit_weight = unit_weight
        self.class_a_cutoff = class_a_cutoff
        self.class_b_cutoff = class_b_cutoff
        
        # Validate weights sum to 1 (100%)
        # np.isclose is used instead of == because of floating point precision issues
        if not np.isclose(order_weight + unit_weight, 1.0):
            raise ValueError("Order weight and unit weight must sum to 1.0")
        
        # Validate cutoffs are in correct order (A < B < 1)
        if not 0 < class_a_cutoff < class_b_cutoff < 1:
            raise ValueError("Cutoffs must satisfy: 0 < class_a_cutoff < class_b_cutoff < 1")
    
    def calculate_score(self, df, sku_column, order_column, unit_column):
        """
        Calculate the combined score for each SKU.
        
        Args:
            df (pandas.DataFrame): DataFrame containing SKU data
            sku_column (str): Name of the column containing SKU identifiers
            order_column (str): Name of the column containing order counts
            unit_column (str): Name of the column containing unit counts
            
        Returns:
            pandas.DataFrame: DataFrame with original data and added score column
        """
        # === BEGINNER'S NOTE ===
        # This function calculates an importance score for each SKU by:
        # 1. Normalizing the order counts and unit counts (converting to 0-1 scale)
        # 2. Applying weights to each factor (order_weight and unit_weight)
        # 3. Adding them together to get a final score
        # Higher scores mean more important items
        # =====================
        
        # Create a copy to avoid modifying the original DataFrame
        # This is a good practice in pandas to prevent unexpected changes to the original data
        result_df = df.copy()
        
        # Normalize the order and unit columns (0-1 scale)
        # This converts the raw numbers to a scale where the highest value = 1
        max_orders = df[order_column].max()
        max_units = df[unit_column].max()
        
        # Avoid division by zero (which would cause errors)
        # If max_orders is 0, all normalized_orders will be 0
        if max_orders > 0:
            normalized_orders = df[order_column] / max_orders
        else:
            normalized_orders = df[order_column] * 0
            
        if max_units > 0:
            normalized_units = df[unit_column] / max_units
        else:
            normalized_units = df[unit_column] * 0
        
        # Calculate the combined score using the weighted average formula
        # Example: If order_weight=0.7 and unit_weight=0.3:
        # score = 0.7 * normalized_orders + 0.3 * normalized_units
        result_df['score'] = (
            self.order_weight * normalized_orders + 
            self.unit_weight * normalized_units
        )
        
        return result_df
    
    def classify(self, df, sku_column, order_column, unit_column):
        """
        Perform ABC classification on the given DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame containing SKU data
            sku_column (str): Name of the column containing SKU identifiers
            order_column (str): Name of the column containing order counts
            unit_column (str): Name of the column containing unit counts
            
        Returns:
            pandas.DataFrame: DataFrame with original data and added ABC classification column
        """
        # === BEGINNER'S NOTE ===
        # This is the main function that performs the ABC classification.
        # The process works like this:
        # 1. Calculate a score for each SKU
        # 2. Sort SKUs by score (highest to lowest)
        # 3. Calculate the cumulative percentage of the total score
        # 4. Assign classes based on cutoff values:
        #    - Class A: Items up to class_a_cutoff (e.g., first 80%)
        #    - Class B: Items between class_a_cutoff and class_b_cutoff (e.g., next 15%)
        #    - Class C: Remaining items (e.g., last 5%)
        # =====================
        
        # Store original columns to preserve them in the result
        original_columns = df.columns.tolist()
        
        # Calculate scores for each SKU using the method we defined earlier
        result_df = self.calculate_score(df, sku_column, order_column, unit_column)
        
        # Sort by score in descending order (highest scores first)
        # reset_index(drop=True) renumbers the rows from 0,1,2,...
        result_df = result_df.sort_values('score', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative percentage (running total)
        total_score = result_df['score'].sum()
        
        if total_score > 0:
            # cumsum() calculates the cumulative sum (running total)
            result_df['cum_score'] = result_df['score'].cumsum()
            # Calculate what percentage of the total each running total represents
            result_df['cum_percentage'] = result_df['cum_score'] / total_score
        else:
            # Handle case where all scores are 0
            result_df['cum_score'] = 0
            result_df['cum_percentage'] = 0
        
        # Assign ABC classes based on cumulative percentage
        # np.select works like a series of if/elif/else statements:
        # - If cum_percentage <= class_a_cutoff, assign 'A'
        # - Else if cum_percentage <= class_b_cutoff, assign 'B'
        # - Otherwise (default), assign 'C'
        conditions = [
            (result_df['cum_percentage'] <= self.class_a_cutoff),
            (result_df['cum_percentage'] <= self.class_b_cutoff)
        ]
        choices = ['A', 'B']
        result_df['abc_class'] = np.select(conditions, choices, default='C')
        
        # Remove the temporary columns we created for calculations
        result_df = result_df.drop(['cum_score', 'cum_percentage'], axis=1)
        
        # Make sure we keep all the original columns from the input dataframe
        for col in original_columns:
            if col not in result_df.columns and col in df.columns:
                result_df[col] = df[col]
        
        return result_df
    
    def process_file(self, input_file, output_file, sku_column, order_column, unit_column):
        """
        Process a CSV file and save the results with ABC classification.
        
        Args:
            input_file (str): Path to the input CSV file
            output_file (str): Path to save the output CSV file
            sku_column (str): Name of the column containing SKU identifiers
            order_column (str): Name of the column containing order counts
            unit_column (str): Name of the column containing unit counts
            
        Returns:
            pandas.DataFrame: The processed DataFrame with ABC classification
        """
        # === BEGINNER'S NOTE ===
        # This function handles the file I/O (input/output) operations:
        # 1. Reading data from a CSV file
        # 2. Detecting the separator (comma or semicolon)
        # 3. Validating the data has the required columns
        # 4. Performing the classification
        # 5. Saving the results to a new CSV file
        # =====================
        
        try:
            # Try to detect the separator (comma or semicolon)
            # This makes the program more flexible with different CSV formats
            with open(input_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if ';' in first_line:
                    separator = ';'
                else:
                    separator = ','
            
            # Read the input file with the detected separator
            df = pd.read_csv(input_file, sep=separator)
            
            # Validate required columns exist
            # This prevents errors later if the CSV is missing important columns
            required_columns = [sku_column, order_column, unit_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Perform classification using the method we defined earlier
            result_df = self.classify(df, sku_column, order_column, unit_column)
            
            # Save to output file with the same separator as input
            # This preserves the original format (comma or semicolon)
            result_df.to_csv(output_file, index=False, sep=separator)
            
            return result_df
            
        except Exception as e:
            # If anything goes wrong, raise an exception with a helpful message
            raise Exception(f"Error processing file: {str(e)}")


def classify_abc(input_file, output_file, sku_column, order_column, unit_column,
                order_weight=0.7, unit_weight=0.3, class_a_cutoff=0.8, class_b_cutoff=0.95):
    """
    Convenience function to perform ABC classification on a file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        sku_column (str): Name of the column containing SKU identifiers
        order_column (str): Name of the column containing order counts
        unit_column (str): Name of the column containing unit counts
        order_weight (float): Weight given to order frequency (default: 0.7)
        unit_weight (float): Weight given to unit volume (default: 0.3)
        class_a_cutoff (float): Cumulative percentage cutoff for Class A (default: 0.8)
        class_b_cutoff (float): Cumulative percentage cutoff for Class B (default: 0.95)
        
    Returns:
        pandas.DataFrame: The processed DataFrame with ABC classification
    """
    # === BEGINNER'S NOTE ===
    # This is a helper function that makes it easier to use the ABCClassifier.
    # Instead of having to create an ABCClassifier object and then call methods on it,
    # you can just call this function directly with all the parameters.
    # 
    # This is the function that the Streamlit app uses to perform ABC classification.
    # =====================
    
    # Create an instance of the ABCClassifier with the specified parameters
    classifier = ABCClassifier(
        order_weight=order_weight,
        unit_weight=unit_weight,
        class_a_cutoff=class_a_cutoff,
        class_b_cutoff=class_b_cutoff
    )
    
    # Use the classifier to process the file
    return classifier.process_file(
        input_file=input_file,
        output_file=output_file,
        sku_column=sku_column,
        order_column=order_column,
        unit_column=unit_column
    )


if __name__ == "__main__":
    # === BEGINNER'S NOTE ===
    # This section runs only when you execute this file directly (not when imported).
    # It provides a command-line interface to run ABC classification from the terminal.
    # 
    # Example command:
    # python abc_classes.py input.csv output.csv --order-weight 0.6 --unit-weight 0.4
    # =====================
    
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Perform ABC classification on SKU data')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('output_file', help='Path to output CSV file')
    parser.add_argument('--sku-column', default='sku', help='Name of SKU column')
    parser.add_argument('--order-column', default='orders', help='Name of orders column')
    parser.add_argument('--unit-column', default='units', help='Name of units column')
    parser.add_argument('--order-weight', type=float, default=0.7, help='Weight for order frequency (0-1)')
    parser.add_argument('--unit-weight', type=float, default=0.3, help='Weight for unit volume (0-1)')
    parser.add_argument('--class-a-cutoff', type=float, default=0.8, help='Cutoff for Class A (0-1)')
    parser.add_argument('--class-b-cutoff', type=float, default=0.95, help='Cutoff for Class B (0-1)')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Validate weights sum to 1
    if not np.isclose(args.order_weight + args.unit_weight, 1.0):
        parser.error("Order weight and unit weight must sum to 1.0")
    
    # Run classification with the provided arguments
    classify_abc(
        input_file=args.input_file,
        output_file=args.output_file,
        sku_column=args.sku_column,
        order_column=args.order_column,
        unit_column=args.unit_column,
        order_weight=args.order_weight,
        unit_weight=args.unit_weight,
        class_a_cutoff=args.class_a_cutoff,
        class_b_cutoff=args.class_b_cutoff
    )
    
    # Print a success message
    print(f"ABC classification complete. Results saved to {args.output_file}")
