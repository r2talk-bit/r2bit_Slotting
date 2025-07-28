"""
Streamlit application for the R2Bit Slotting project.
This is the main entry point for the web interface.
"""

import streamlit as st
import pandas as pd
import os
import tempfile
import matplotlib.pyplot as plt
from src.abc_classes import classify_abc
from src.slotting import perform_slotting

# Set up the page
st.set_page_config(
    page_title="R2Bit Slotting Manager",
    page_icon="🏭",
    layout="wide"
)

st.title("R2Bit Warehouse Slotting")
st.sidebar.header("Configuration")

# File upload
st.sidebar.subheader("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Try to detect the separator (comma or semicolon)
    content = uploaded_file.getvalue().decode('utf-8')
    first_line = content.split('\n')[0]
    if ';' in first_line:
        separator = ';'
    else:
        separator = ','
    
    # Read the uploaded file
    df = pd.read_csv(uploaded_file, sep=separator)
    
    # Display raw data
    with st.expander("Raw Data"):
        st.dataframe(df)
    
    # ABC Classification Parameters
    st.sidebar.subheader("2. ABC Classification Parameters")
    
    # Get column names from the dataframe
    columns = df.columns.tolist()
    
    # Select columns for classification
    sku_column = st.sidebar.selectbox("SKU Column", columns, index=columns.index('sku') if 'sku' in columns else 0)
    order_column = st.sidebar.selectbox("Order Frequency Column", columns, index=columns.index('orders') if 'orders' in columns else 0)
    unit_column = st.sidebar.selectbox("Unit Volume Column", columns, index=columns.index('units') if 'units' in columns else 0)
    
    # Classification weights and cutoffs
    col1, col2 = st.sidebar.columns(2)
    with col1:
        order_weight = st.number_input("Order Weight", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    with col2:
        unit_weight = st.number_input("Unit Weight", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        class_a_cutoff = st.number_input("Class A Cutoff", min_value=0.1, max_value=0.9, value=0.8, step=0.05)
    with col2:
        class_b_cutoff = st.number_input("Class B Cutoff", min_value=0.1, max_value=0.99, value=0.95, step=0.05)
    
    # Warehouse Layout Parameters
    st.sidebar.subheader("3. Warehouse Layout")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        num_ruas = st.number_input("Number of Rows", min_value=1, value=10, step=1)
        num_blocos = st.number_input("Blocks per Row", min_value=1, value=10, step=1)
    with col2:
        num_niveis = st.number_input("Levels per Block", min_value=1, value=5, step=1)
        num_apartamentos = st.number_input("Apartments per Level", min_value=1, value=10, step=1)
    
    # Zone Definition Parameters
    st.sidebar.subheader("4. Zone Definition")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        zona_a_ruas = st.number_input("Zone A Rows", min_value=1, max_value=num_ruas, value=3, step=1)
        zona_a_niveis = st.number_input("Zone A Levels", min_value=1, max_value=num_niveis, value=2, step=1)
        zona_a_blocos = st.number_input("Zone A Blocks", min_value=1, max_value=num_blocos, value=3, step=1)
    with col2:
        zona_b_percentage = st.number_input("Zone B Percentage", min_value=0.1, max_value=0.9, value=0.3, step=0.05, format="%.2f")
    
    # Run Slotting button
    if st.sidebar.button("Run Slotting", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            # Create temporary files for intermediate results
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_input_file:
                temp_input_path = temp_input_file.name
                df.to_csv(temp_input_path, sep=separator, index=False)
            
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_abc_file:
                temp_abc_path = temp_abc_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_slotting_file:
                temp_slotting_path = temp_slotting_file.name
            
            try:
                # Run ABC Classification
                abc_df = classify_abc(
                    input_file=temp_input_path,
                    output_file=temp_abc_path,
                    sku_column=sku_column,
                    order_column=order_column,
                    unit_column=unit_column,
                    order_weight=order_weight,
                    unit_weight=unit_weight,
                    class_a_cutoff=class_a_cutoff,
                    class_b_cutoff=class_b_cutoff
                )
                
                # Run Slotting
                slotting_df = perform_slotting(
                    input_file=temp_abc_path,
                    output_file=temp_slotting_path,
                    num_ruas=num_ruas,
                    num_blocos=num_blocos,
                    num_niveis=num_niveis,
                    num_apartamentos=num_apartamentos,
                    zona_a_ruas=zona_a_ruas,
                    zona_a_blocos=zona_a_blocos,
                    zona_a_niveis=zona_a_niveis,
                    zona_b_percentage=zona_b_percentage
                )
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["ABC Classification", "Slotting Results", "Warehouse Statistics"])
                
                with tab1:
                    st.header("ABC Classification Results")
                    
                    # Count SKUs by class
                    class_counts = abc_df['abc_class'].value_counts().reset_index()
                    class_counts.columns = ['Class', 'Count']
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader("Class Distribution")
                        st.dataframe(class_counts)
                        
                        # Create a pie chart using matplotlib
                        fig, ax = plt.subplots()
                        ax.pie(class_counts['Count'], labels=class_counts['Class'], autopct='%1.1f%%')
                        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("ABC Classification Details")
                        st.dataframe(abc_df)
                
                with tab2:
                    st.header("Slotting Results")
                    
                    # Count SKUs by zone
                    zone_counts = slotting_df['zone'].value_counts().reset_index()
                    zone_counts.columns = ['Zone', 'Count']
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader("Zone Distribution")
                        st.dataframe(zone_counts)
                        
                        # Create a simple pie chart
                        st.subheader("Zone Distribution")
                        fig, ax = plt.subplots()
                        ax.pie(zone_counts['Count'], labels=zone_counts['Zone'], autopct='%1.1f%%')
                        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                        st.pyplot(fig)
                        
                        # Cross-tabulation of ABC class vs Zone
                        st.subheader("Class vs Zone")
                        cross_tab = pd.crosstab(slotting_df['abc_class'], slotting_df['zone'])
                        st.dataframe(cross_tab)
                    
                    with col2:
                        st.subheader("Slotting Details")
                        st.dataframe(slotting_df)
                
                with tab3:
                    st.header("Warehouse Statistics")
                    
                    # Calculate warehouse statistics
                    total_locations = num_ruas * num_blocos * num_niveis * num_apartamentos
                    zone_a_locations = zona_a_ruas * zona_a_blocos * zona_a_niveis * num_apartamentos
                    non_zone_a = total_locations - zone_a_locations
                    zone_b_locations = int(non_zone_a * zona_b_percentage)
                    zone_c_locations = non_zone_a - zone_b_locations
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Warehouse Layout")
                        st.info(f"Total Locations: {total_locations}")
                        st.info(f"Zone A Locations: {zone_a_locations} ({zone_a_locations/total_locations:.1%})")
                        st.info(f"Zone B Locations: {zone_b_locations} ({zone_b_locations/total_locations:.1%})")
                        st.info(f"Zone C Locations: {zone_c_locations} ({zone_c_locations/total_locations:.1%})")
                    
                    with col2:
                        st.subheader("Utilization")
                        assigned_locations = len(slotting_df[slotting_df['location'] != "Not assigned"])
                        utilization = assigned_locations / total_locations
                        
                        st.info(f"Assigned Locations: {assigned_locations}")
                        st.info(f"Warehouse Utilization: {utilization:.1%}")
                        
                        # Utilization by zone
                        zone_a_used = len(slotting_df[slotting_df['zone'] == 'A'])
                        zone_b_used = len(slotting_df[slotting_df['zone'] == 'B'])
                        zone_c_used = len(slotting_df[slotting_df['zone'] == 'C'])
                        
                        st.info(f"Zone A Utilization: {zone_a_used}/{zone_a_locations} ({zone_a_used/zone_a_locations:.1%})")
                        st.info(f"Zone B Utilization: {zone_b_used}/{zone_b_locations} ({zone_b_used/zone_b_locations:.1%})")
                        st.info(f"Zone C Utilization: {zone_c_used}/{zone_c_locations} ({zone_c_used/zone_c_locations:.1%})")
                
                # Download buttons for results
                col1, col2 = st.columns(2)
                with col1:
                    abc_csv = abc_df.to_csv(sep=separator, index=False)
                    st.download_button(
                        label="Download ABC Classification Results",
                        data=abc_csv,
                        file_name="abc_classification_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    slotting_csv = slotting_df.to_csv(sep=separator, index=False)
                    st.download_button(
                        label="Download Slotting Results",
                        data=slotting_csv,
                        file_name="warehouse_slotting_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            finally:
                # Clean up temporary files
                for temp_file in [temp_input_path, temp_abc_path, temp_slotting_path]:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
else:
    st.info("Please upload a CSV file to get started.")
    st.markdown("""
    ### Expected CSV Format
    
    Your CSV file should contain at least the following columns:
    - SKU identifier
    - Order frequency (number of orders)
    - Unit volume (number of units)
    
    Example:
    ```
    sku;orders;units;description
    SKU001;120;500;"High frequency item"
    SKU002;80;1200;"High volume item"
    ```
    """)

# Add a footer
st.sidebar.markdown("---")
st.sidebar.info("R2Bit Slotting Manager v1.0")
