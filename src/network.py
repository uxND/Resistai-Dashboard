import networkx as nx
from pyvis.network import Network
import pandas as pd
import os

def load_card_data(filepath="data/card_annotations.tsv"):
    """
    Attempts to load the CARD database annotations.
    If the file is missing or empty, it provides a realistic default dataset 
    for demonstration purposes.
    """
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            # Assuming tab-separated as per standard bioinformatics databases
            df = pd.read_csv(filepath, sep='\t')
            # Ensure required columns exist
            if all(col in df.columns for col in ['gene_name', 'antibiotic', 'gene_family']):
                return df
    except Exception as e:
        print(f"Error reading CARD data: {e}")

    print("⚠️ Valid CARD database not found or incomplete. Generating realistic demo network data...")
    # High-quality mock data representing real-world AMR mechanisms
    demo_data = {
        'gene_name': [
            'blaKPC-2', 'blaNDM-1', 'mecA', 'vanA', 'vanB', 
            'tet(M)', 'erm(B)', 'mcr-1', 'qnrA', 'aac(6\')-Ib-cr'
        ],
        'antibiotic': [
            'Meropenem', 'Imipenem', 'Oxacillin', 'Vancomycin', 'Vancomycin',
            'Tetracycline', 'Erythromycin', 'Colistin', 'Ciprofloxacin', 'Ciprofloxacin'
        ],
        'gene_family': [
            'Class A beta-lactamase', 'Metallo-beta-lactamase', 'PBP2a', 'D-Ala-D-Lac ligase', 'D-Ala-D-Lac ligase',
            'Ribosomal protection', 'rRNA methylase', 'Phosphoethanolamine transferase', 'Target protection', 'Aminoglycoside acetyltransferase'
        ]
    }
    return pd.DataFrame(demo_data)

def build_gene_network():
    """
    Builds a force-directed graph mapping resistance genes to the antibiotics they defeat.
    Saves the interactive visualization as an HTML file for Streamlit to render.
    """
    print("🕸️ Building Resistance Gene Network...")
    
    card_df = load_card_data()
    G = nx.Graph()

    # Build the network nodes and edges
    for _, row in card_df.iterrows():
        gene = row['gene_name']
        drug = row['antibiotic']
        family = row['gene_family']

        # Add Gene Node (Group 1)
        G.add_node(gene, type='gene', title=f"Gene Family: {family}", group=1)
        
        # Add Antibiotic Node (Group 2)
        G.add_node(drug, type='antibiotic', title="Antibiotic Class", group=2)
        
        # Add Edge linking the gene to the antibiotic it resists
        G.add_edge(gene, drug, weight=2)

    # Initialize Pyvis Network
    # Using a white background and dark text to match the Streamlit light/dark modes gracefully
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.from_nx(G)

    # Apply styling
    for node in net.nodes:
        if node.get('group') == 1:
            node['color'] = '#7F77DD'  # Purple for Genes
            node['size'] = 25
            node['shape'] = 'dot'
        else:
            node['color'] = '#1D9E75'  # Teal for Antibiotics
            node['size'] = 35
            node['shape'] = 'square'

    # Configure physics for a beautiful, stable layout
    net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4
      },
      "edges": {
        "color": {"inherit": true},
        "smooth": {"type": "continuous"}
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "centralGravity": 0.01,
          "springLength": 150,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 150}
      }
    }
    """)
    
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    
    # Save the graph
    output_path = "outputs/resistance_network.html"
    net.write_html(output_path)
    print(f"✅ Network visualization successfully saved to {output_path}")

if __name__ == "__main__":
    # Execute the build function when the script is run directly
    build_gene_network()