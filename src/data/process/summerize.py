from transformers import BartTokenizer, BartForConditionalGeneration
import torch

model_path = '/home/fulian/RAG/summerize_model'
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def summarize_batch(articles: list[str], batch_size: int = 4):
    """
    Takes a list of articles and returns a list of summaries.
    """
    summaries = []
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        inputs = tokenizer(
            batch, 
            max_length=3000, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(device)

        # 3. Generate Summaries for the whole batch
        summary_ids = model.generate(
            inputs["input_ids"], 
            num_beams=4, 
            max_length=250, 
            early_stopping=True
        )

        # 4. Decode and clean up
        batch_summaries = tokenizer.batch_decode(
            summary_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        summaries.extend(batch_summaries)
        print(f"✅ Processed {len(summaries)}/{len(articles)} articles...")

    return summaries

