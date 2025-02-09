import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

########################################
# 1. Load the fine-tuned model & tokenizer
########################################

model_path = "./models/pc-aspirational-model-2"  # or your final checkpoint path
model_checkpoint = "google/flan-t5-base"  # or "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

########################################
# 2. Inference on test data
########################################

test_file = "aspirational_test2.jsonl"
output_file = "test_predictions2.json"  # We'll store results here

results = []

with open(test_file, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, 1):
        data = json.loads(line)
        prompt_text = data["prompt"]
        gold_completion = data["completion"]  # Ground truth

        # Tokenize & generate multiple predictions
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=5,
        num_return_sequences=1   # Only one prediction is generated
         )

        # Decode all generated sequences
        pred_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Print to console
        print(f"\n=== Example {line_idx} ===")
        print("Prompt:\n", prompt_text)
        for idx, text in enumerate(pred_texts, start=1):
            print(f"Prediction {idx}:\n{text}")
        print("Gold:\n", gold_completion)

        # Save results (storing multiple predictions)
        results.append({
            "example_id": line_idx,
            "prompt": prompt_text,
            "gold": gold_completion,
            "predictions": pred_texts
        })

########################################
# 3. Save predictions to JSON file
########################################

with open(output_file, "w", encoding="utf-8") as out:
    json.dump(results, out, ensure_ascii=False, indent=2)

print(f"\nInference complete. Results saved to {output_file}.")
