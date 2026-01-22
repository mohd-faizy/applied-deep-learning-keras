import json
import os

file_path = 'e:/001_Github_Repo_all/applied-deep-learning-keras/06_Embeddings.ipynb'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Validate Cell 1
    # Check if Cell 1 is a code cell
    if notebook['cells'][1]['cell_type'] != 'code':
        print(f"Cell 1 is not a code cell. It is {notebook['cells'][1]['cell_type']}")
        # Maybe the cells are shifted?
        # Let's search for the cell with 'n_teams = 10887' or similar
        found_idx = -1
        for i, cell in enumerate(notebook['cells']):
             if cell['cell_type'] == 'code':
                 src = "".join(cell['source'])
                 if "Embedding" in src and "Input" in src: # Generic check
                     found_idx = i
                     break
        if found_idx != -1:
            print(f"Found target code cell at index {found_idx}")
            idx = found_idx
        else:
            print("Could not find target code cell.")
            exit(1)
    else:
        idx = 1
        print("Cell 1 is code.")

    # Show what we found
    current_source = "".join(notebook['cells'][idx]['source'])
    print(f"Current source (first 100 chars): {current_source[:100]}")

    # Force update if it looks like the right cell (has Embedding/Input)
    # AND doesn't already have 'num_unique_teams' (to avoid double update if I ran it before?)
    if "num_unique_teams" in current_source:
         print("Already updated?")
    else:
        new_source_code = [
            "import tensorflow as tf\n",
            "from tensorflow.keras.layers import Embedding, Input, Flatten\n",
            "from tensorflow.keras.models import Model\n",
            "\n",
            "# Define the number of unique categories\n",
            "num_unique_teams = 10887\n",
            "\n",
            "# Input Layer: Expects a single integer (Team ID)\n",
            "input_tensor = Input(shape=(1,), name='Team_ID_Input')\n",
            "\n",
            "# Embedding Layer: Maps integer -> float\n",
            "# input_dim needs to match the number of unique categories\n",
            "embed_layer = Embedding(input_dim=num_unique_teams,\n",
            "                        output_dim=1,\n",
            "                        name='Team-Strength-Lookup')\n",
            "\n",
            "# Connect Input to Embedding\n",
            "embed_tensor = embed_layer(input_tensor)\n",
            "\n",
            "# Flatten Layer: Reshapes (Batch, 1, 1) -> (Batch, 1)\n",
            "flatten_tensor = Flatten(name='Flatten_Output')(embed_tensor)\n",
            "\n",
            "# Build the Model\n",
            "model = Model(inputs=input_tensor, outputs=flatten_tensor)\n",
            "\n",
            "# Display Summary\n",
            "model.summary()"
        ]
        notebook['cells'][idx]['source'] = new_source_code
        print("Updated code cell content.")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

    print("Notebook saved.")

except Exception as e:
    print(f"Error: {e}")
