# text_embedding.py

import torch
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class TextEmbedder:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def embed_texts(self, texts, max_len=128, batch_size=32):
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                tokens = self.tokenizer(
                    batch,
                    max_length=max_len,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                output = self.model(**tokens)
                cls_embeddings = output.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def run(self, input_csv_path, output_dir="data/processed"):
        df = pd.read_csv(input_csv_path)
        texts = df["combined_text"].astype(str).tolist()
        embeddings = self.embed_texts(texts)

        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f"text_emb_{i}" for i in range(embeddings.shape[1])]
        )

        result_df = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"text_embeddings_{timestamp}.csv"
        result_df.to_csv(output_path, index=False)

        print(f"✅ Text embeddings saved: {output_path}")
        return output_path
