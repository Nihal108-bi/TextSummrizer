from src.TextSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        # Initialize tokenizer and model once to save resources
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)

    def predict(self, text):
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

        print("Dialogue:")
        print(text)

        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
        
        # Generate summary IDs
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            **gen_kwargs
        )

        # Decode summary IDs to text
        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        print("\nModel Summary:")
        print(output)

        return output
