"""
ABC Classification module for the R2Bit Slotting project.
This module provides functionality to classify SKUs using the ABC method
based on order frequency and unit volume.
"""

import pandas as pd
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
        self.order_weight = order_weight
        self.unit_weight = unit_weight
        self.class_a_cutoff = class_a_cutoff
        self.class_b_cutoff = class_b_cutoff
        
        # Validate weights sum to 1
        if not np.isclose(order_weight + unit_weight, 1.0):
            raise ValueError("Order weight and unit weight must sum to 1.0")
        
        # Validate cutoffs
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
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Normalize the order and unit columns (0-1 scale)
        max_orders = df[order_column].max()
        max_units = df[unit_column].max()
        
        # Avoid division by zero
        if max_orders > 0:
            normalized_orders = df[order_column] / max_orders
        else:
            normalized_orders = df[order_column] * 0
            
        if max_units > 0:
            normalized_units = df[unit_column] / max_units
        else:
            normalized_units = df[unit_column] * 0
        
        # Calculate the combined score
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
        # Calculate scores
        result_df = self.calculate_score(df, sku_column, order_column, unit_column)
        
        # Sort by score in descending order
        result_df = result_df.sort_values('score', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative percentage
        total_score = result_df['score'].sum()
        
        if total_score > 0:
            result_df['cum_score'] = result_df['score'].cumsum()
            result_df['cum_percentage'] = result_df['cum_score'] / total_score
        else:
            # Handle case where all scores are 0
            result_df['cum_score'] = 0
            result_df['cum_percentage'] = 0
        
        # Assign ABC classes based on cumulative percentage
        conditions = [
            (result_df['cum_percentage'] <= self.class_a_cutoff),
            (result_df['cum_percentage'] <= self.class_b_cutoff)
        ]
        choices = ['A', 'B']
        result_df['abc_class'] = np.select(conditions, choices, default='C')
        
        # Clean up intermediate columns if needed
        result_df = result_df.drop(['cum_score', 'cum_percentage'], axis=1)
        
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
            required_columns = [sku_column, order_column, unit_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Perform classification
            result_df = self.classify(df, sku_column, order_column, unit_column)
            
            # Save to output file with the same separator as input
            result_df.to_csv(output_file, index=False, sep=separator)
            
            return result_df
            
        except Exception as e:
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
    classifier = ABCClassifier(
        order_weight=order_weight,
        unit_weight=unit_weight,
        class_a_cutoff=class_a_cutoff,
        class_b_cutoff=class_b_cutoff
    )
    
    return classifier.process_file(
        input_file=input_file,
        output_file=output_file,
        sku_column=sku_column,
        order_column=order_column,
        unit_column=unit_column
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
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
    
    args = parser.parse_args()
    
    # Validate weights sum to 1
    if not np.isclose(args.order_weight + args.unit_weight, 1.0):
        parser.error("Order weight and unit weight must sum to 1.0")
    
    # Run classification
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
    
    print(f"ABC classification complete. Results saved to {args.output_file}")
