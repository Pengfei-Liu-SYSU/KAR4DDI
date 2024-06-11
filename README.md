# Knowledge Augmentation with Reinforcement Learning for Drug-Drug Interaction Event Few-shot Predictions (KAR4DDI)

Drug-drug interaction events (DDIE) prediction is crucial for preventing adverse reactions and ensuring optimal therapeutic outcomes. However, existing methods often face challenges with imbalanced datasets, complex interaction mechanisms, and poor generalization to unknown drug combinations. To address these challenges, we propose a pioneering knowledge augmentation framework that adaptively infuses prior drug knowledge into a large language model (LLM). This framework employs reinforcement learning (RL) to refine the process of knowledge extraction and synthesis, significantly reducing the knowledge and parameter exploration space and enhancing the accuracy of the LLM for DDIE predictions.

### Dataset Download links
- [Dataset](https://huggingface.co/datasets/liupf/KAR4DDI)

## Code Structure

### `ckpts`
Directory containing checkpoints used for different purposes:
- **finetune_ckpts**: Checkpoints from fine-tuning processes.
- **text_ckpts**: Contains the checkpoints related to text models.
    - [BioT5](https://huggingface.co/QizhiPei/biot5-base): A model checkpoint for text and chemical data integration.

### `data/deepddie`
This directory includes materials for working with molecular data:
- **drug_cluster.csv**: Contains PCA features and results of k-means, Birch, and Agglomerative clustering of drugs.
- **drugbank_id_smiles.txt**: Contains mappings of drug `drugbank_id` to their corresponding SMILES (Simplified Molecular Input Line Entry System) representations.
- **smiles_description.csv**: Includes the SMILES representation of drugs along with their descriptions.
- **few_k_b_a_5_20**: Contains a dataset for few-shot learning in DDIE (Drug-Drug Interaction Extraction), split into the proportions 2:2:6.

### `src`
Source code directory housing the implementation details:
- **`datasets`**: Code for constructing datasets.
    - `dataset_manager.py`: Creation of datasets for adaptive knowledge injection.
- **`models`**: Core models for the tasks.
    - `init.py`: Initializes model parameters and settings.
    - `model_manager.py`: Manages the loading and handling of models for knowledge injection.
    - `BioT5.py`: BioT5.
- **`utils`**: Utility functions and initializations.
    - `init.py`: General utility tool initialization.
    - `xutils.py`: Advanced and specialized utility tool initialization.
- **`task_manager.py`**: Function to execute tasks related to adaptive knowledge augmentation.

### Detailed Parameter Explanations for Tasks
- `script_mode`: Select the operation mode. Options include `train_rl`, `data_check`, `encoder_check`, `train`, and `eval`.
- `shot`: Select the data split mode. Options include `rare`, `few`, `com`, and `all`.

## Acknowledgments

The development of the dataset was greatly inspired by the DeepDDI 2.
