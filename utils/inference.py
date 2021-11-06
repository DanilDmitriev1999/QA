import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def classification(context, question, model, tokenizer, device=device):
    encodings = tokenizer(context, question, truncation=True, return_tensors="pt")
    text = encodings['input_ids'].to(device)
    mask = encodings['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(text, mask)
    
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)
    
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    start_pred = torch.argmax(start_logits, dim=1).squeeze(-1).cpu().detach().numpy()

    end_pred = torch.argmax(end_logits, dim=1).squeeze(-1).cpu().detach().numpy()
    print(f"predict: {tokenizer.decode(text[0][start_pred: end_pred+1])}")

if __name__ == '__main__':
    pass