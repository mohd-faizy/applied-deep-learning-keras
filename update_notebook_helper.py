import json
import os

file_path = 'e:/001_Github_Repo_all/applied-deep-learning-keras/06_Embeddings.ipynb'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Update Cell 1 (Code) - Index 1
    # Check if this is indeed the correct cell by looking at the content
    if "n_teams = 10887" in "".join(notebook['cells'][1]['source']):
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
        notebook['cells'][1]['source'] = new_source_code
        print("Updated code cell.")
    else:
        print("Could not find expected code in cell 1. Skipping code update.")

    # Update Cell 2 (Markdown) - Index 2
    if "Detailed Explanation of Arguments" in "".join(notebook['cells'][2]['source']):
        new_source_md = [
            "\n",
            "### Detailed Explanation of Arguments\n",
            "| Argument | Value | Description |\n",
            "| :--- | :--- | :--- |\n",
            "| `input_dim` | `num_unique_teams` (10887) | The size of the vocabulary (number of unique categories). |\n",
            "| `output_dim` | `1` | The size of the vector space for the embedding. |\n",
            "| `name` | `'Team-Strength-Lookup'` | A custom name for the layer, useful for debugging and visualization. |"
        ]
        notebook['cells'][2]['source'] = new_source_md
        print("Updated markdown cell.")
    else:
        print("Could not find expected markdown in cell 2. Skipping markdown update.")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2) # Keep indentation consistent

    print("Notebook updated successfully.")

except Exception as e:
    print(f"Error: {e}")
