import os
from typing import Dict, List

import json
from openai import OpenAI

CACHE_PATH = './embeddings.json'

class GetOpenAIEmbeddings:
    def __init__(self, openai_session: OpenAI, model: str = 'text-embedding-3-large'):
        self.model = model
        self.cache = None
        self.openai_session = openai_session

    def _get_embedding_from_cache(self, text: str) -> str:
        if self.cache is None:
            if os.path.exists(CACHE_PATH):
                with open(CACHE_PATH, 'r') as openfile:
                    cache = json.load(openfile)
                    self.cache = cache

        if self.model in self.cache and text in self.cache[self.model]:
            return self.cache[self.model][text]
        
    def _save_embeddings_to_cache(self, embeds_dict: Dict[str, str]):                         
        if self.cache is None:
            if os.path.exists(CACHE_PATH):
                with open(CACHE_PATH, 'r') as openfile:
                    cache = json.load(openfile)
                    self.cache = cache    

        self.cache[self.model] = embeds_dict | self.cache[self.model]

        with open(CACHE_PATH, 'w') as openfile:
            json.dump(self.cache, openfile)        
        
    def _get_embedding_from_openai(self, texts: List[str]) -> List:   
        response = self.openai_session.embeddings.create(
                        model = self.model,
                        input = texts
        )

        return response.data #[0].embedding

    def get_embedding(self, texts: List[str]) -> Dict[str, str]:
        result_dict = {}

        not_cached_embs = []

        for txt in texts:
            cached_emb = self._get_embedding_from_cache(txt)

            if cached_emb is not None:
                result_dict[txt] = cached_emb
            else:
                not_cached_embs.append(txt)
        
        embeddings = [ x.embedding for x in self._get_embedding_from_openai(not_cached_embs) ]

        for emb in embeddings:
            result_dict[txt] = emb

        self._save_embeddings_to_cache(result_dict)

        return result_dict