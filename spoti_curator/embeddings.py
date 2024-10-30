import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Optional, Union
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """Dataset class for texts"""
    def __init__(self, texts: List[str], max_length: int):
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class GenreAnalyzer:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
        batch_size: int = 32
    ):
        """Initialize the genre analyzer with specified model.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cpu' or 'cuda')
            batch_size: Number of texts to process at once
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def _process_batch(self, batch: List[str], max_length: int) -> np.ndarray:
        """Process a batch of texts and return their embeddings."""
        # Tokenize the batch
        inputs = self.tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs[0]

            # Mean pooling
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        return embeddings.cpu().numpy()

    def get_embedding(
        self, 
        texts: Union[str, List[str]], 
        max_length: int = 256
    ) -> Optional[np.ndarray]:
        """Generate embeddings for one or more texts.
        
        Args:
            texts: Single text string or list of text strings
            max_length: Maximum token length for each text
            
        Returns:
            numpy array of embeddings with shape (n_texts, embedding_dim)
            or None if an error occurs
        """
        try:
            # Convert single text to list for uniform processing
            if isinstance(texts, str):
                texts = [texts]

            # Create dataset and dataloader for batch processing
            dataset = TextDataset(texts, max_length)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )

            # Process batches and collect embeddings
            all_embeddings = []
            for batch in dataloader:
                batch_embeddings = self._process_batch(batch, max_length)
                all_embeddings.append(batch_embeddings)

            # Concatenate all embeddings
            if all_embeddings:
                return np.vstack(all_embeddings)
            return None

        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return None

    def __call__(
        self, 
        texts: Union[str, List[str]], 
        max_length: int = 256
    ) -> Optional[np.ndarray]:
        """Convenience method to call get_embedding directly."""
        return self.get_embedding(texts, max_length)