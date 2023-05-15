from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from src.retrivalModel import Head, Retrival_model

if __name__ == '__main__':
    ### Exercise 2: Working with Real LLMs
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    input = tokenizer("War never changes", return_tensors='pt')
    print(input)

    outputs = model.generate(input['input_ids'], max_length=50, attention_mask=input['attention_mask'])
    print("Greedy:", tokenizer.decode(outputs[0]))

    outputs = model.generate(input['input_ids'], max_length=66, do_sample=True, temperature=0.4,
                             attention_mask=input['attention_mask'])
    print("Sampled low temp:", tokenizer.decode(outputs[0]))

    outputs = model.generate(input['input_ids'], max_length=66, do_sample=True, temperature=0.9,
                             attention_mask=input['attention_mask'])
    print("Sampled high temp:", tokenizer.decode(outputs[0]))

    ### Exercise 3.3: Training a Retrieval Model
    save_path = "img"
    data_path = "datasets/scifact"
    model_dim = [768, 300, 128]
    lr = 1e-4
    epochs = 1500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

    downstream_head = Head(model_dim).to(device)

    optimizer = torch.optim.Adam(downstream_head.parameters(), lr=lr)
    sim_fun = torch.mm

    loss_fn = torch.nn.CrossEntropyLoss()

    retrival_model = Retrival_model(downstream_head, model, tokenizer, data_path, optimizer, loss_fn, sim_fun,
                                    save_path, device)

    retrival_model.train(epochs=epochs)
