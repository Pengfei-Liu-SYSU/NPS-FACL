# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)

import warnings

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#from models.chemT5 import ChemT5
from models.model_manager import MainModel
from datasets.dataset_manager import MainDataset
from utils.xutils import print_model_info, custom_collate_fn, ToDevice
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from torch.optim.lr_scheduler import StepLR
from utils import AverageMeter
import datetime
import time
from models import SUPPORTED_Tokenizer, SUPPORTED_CKPT
import pandas as pd
from rdkit import Chem

#torch.cuda.amp.autocast(enabled=True)

def train_mol_decoder(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss = None):
    running_loss = AverageMeter()
    step = 0
    loss_values = {"train_loss": [], "valid_loss":[]}
    last_ckpt_file = None
    patience = 0
    device = args.device
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        #model.train()
        train_loss = []
        train_loader = tqdm(train_loader, desc="Training")
        for mol in train_loader:
            mol = ToDevice(mol, args.device)
            loss, _ = model(mol)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss.update(loss.detach().cpu().item())
            # step += 1
            # if step % args.logging_steps == 0:
            #     logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
            #     train_loss.append(running_loss.get_average())
            #     running_loss.reset()
        train_loss = running_loss.get_average()
        logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
        loss_values["train_loss"].append(running_loss.get_average())
        running_loss.reset()
        valid_loss = val_mol_decoder(valid_loader, model, device)
        loss_values["valid_loss"].append(valid_loss)

        if best_loss == None or valid_loss<best_loss :
            patience = 0
            best_loss = valid_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            if not os.path.exists(f"{args.ckpt_output_path}/{args.model_name}"):
                os.makedirs(f"{args.ckpt_output_path}/{args.model_name}")
  
            ckpt_file = f"{epoch}_{timestamp}.pth"
            ckpt_path = os.path.join(f"{args.ckpt_output_path}/{args.model_name}", ckpt_file)
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss
            }, ckpt_path)
                
            message = f"epoch: {epoch}, best_loss:{best_loss} ,train_loss:{train_loss}, {ckpt_file} saved. "
            print(message)
            # if last_ckpt_file is not None and os.path.exists(last_ckpt_file):
            #     os.remove(last_ckpt_file)
            #     print(f"Deleted checkpoint file: {last_ckpt_file}")
            last_ckpt_file = ckpt_path
            print(loss_values)
        else:
            patience = patience+1
            scheduler.step()
            message = f"epoch: {epoch}, best_loss:{best_loss} ,train_loss:{train_loss}, ckpt passed, patience : {patience}. "
            if last_ckpt_file is not None :
                state_dict = torch.load(last_ckpt_file, map_location='cpu')["model_state_dict"]
                best_loss = torch.load(last_ckpt_file, map_location='cpu')["best_loss"]
                model.load_state_dict(state_dict, strict = False)
            metric, cases = test_mol_decoder(test_loader, model, device, message)
            message = message + f"epoch {epoch-1} metric : {metric}."
            print(message)
            print(cases)
            print(loss_values)
        if epoch>48:
            message = f"epoch: {epoch}, best_loss:{best_loss} ,train_loss:{train_loss}, ckpt passed, patience : {patience}. "
            metric, cases = test_mol_decoder(test_loader, model, device, message)
            message = message + f"epoch {epoch-1} metric : {metric}."
            print(message)
            print(cases)
            print(loss_values)
            print("Early stopping due to reaching epoch limit.")
            break
        if patience > args.patience:
            print("Early stopping due to reaching patience limit.")
            break

def custom_collate_fn(batch):

    smiles = [item["smiles"] for item in batch]  
    batch = {key: torch.stack([item[key] for item in batch]) for key in batch[0] if key != "smiles"}
    batch["smiles"] = smiles  
    
    return batch
    
def val_mol_decoder(test_loader, model, device, message=None):
    model.eval()  
    test_loss = 0
    y_true, y_pred, y_proba = [], [], []


    with torch.no_grad():
        for batch in test_loader:
            
            smiles_list = batch.pop("smiles")  
            
            batch = {key: val.to(device) for key, val in batch.items()}  
            
            loss, logits = model(batch)
    
            test_loss += loss.item()

            probas = torch.sigmoid(logits).cpu().numpy()
            preds = (probas > 0.5).astype(int)
    
       
            y_true.extend(batch["output"].cpu().numpy())
            y_pred.extend(preds)
            y_proba.extend(probas)

    test_loss /= len(test_loader)
    return test_loss
    

def test_mol_decoder(test_loader, model, device, message=None, save_json_path="low_confidence_cases.json"):

    model.eval()
    test_loss = 0
    y_true, y_pred, y_proba = [], [], []
    smiles_list = []

    with torch.no_grad():
        for batch in test_loader:
            smiles_list.extend(batch["smiles"])
            batch.pop("smiles")
            batch = {key: val.to(device) for key, val in batch.items()}
            

            loss, logits = model(batch)
            test_loss += loss.item()


            if logits.dim() > 1 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            probas = torch.sigmoid(logits).cpu().numpy()
            preds = (probas > 0.5).astype(int)

            y_true.extend(batch["output"].cpu().numpy())
            y_pred.extend(preds)
            y_proba.extend(probas)

    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    confidence_cases = list(zip(smiles_list, y_true, y_pred, y_proba))


    fuzzy_cases = sorted(confidence_cases, key=lambda x: abs(x[3] - 0.5))[:20]

    incorrect_cases = [case for case in confidence_cases if case[1] != case[2]]  # y_true != y_pred
    incorrect_low_confidence_cases = sorted(
        incorrect_cases,
        key=lambda x: -max(x[3], 1 - x[3]),  
        reverse=True
    )[:20]

    fuzzy_cases = [
        {
            "smiles": smiles,
            "truth": int(truth),
            "predicted_label": int(label),
            "confidence": float(conf),
            "is_incorrect": int(truth) != int(label)
        }
        for smiles, truth, label, conf in fuzzy_cases
    ]

    incorrect_low_confidence_cases = [
        {
            "smiles": smiles,
            "truth": int(truth),
            "predicted_label": int(label),
            "confidence": float(conf),
            "is_incorrect": True
        }
        for smiles, truth, label, conf in incorrect_low_confidence_cases
    ]

    fuzzy_incorrect_count = sum(1 for case in fuzzy_cases if case["is_incorrect"])
    print(f"\n🔍 Fuzzy Cases (Confidence closest to 0.5): {len(fuzzy_cases)} samples")
    print(f"Number of incorrect predictions in fuzzy cases: {fuzzy_incorrect_count}/{len(fuzzy_cases)}")

    print(f"\n🔍 Incorrect Cases (High confidence but wrong): {len(incorrect_low_confidence_cases)} samples")

    json_output = {
        "fuzzy_cases": fuzzy_cases,
        "incorrect_low_confidence_cases": incorrect_low_confidence_cases
    }
    
    with open(save_json_path, "w") as json_file:
        json.dump(json_output, json_file, indent=4)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    print(f"\n📌 Test Results [{timestamp}] {message if message else ''}")
    print(f"🔹 Test Loss: {test_loss:.4f}")
    print(f"🔹 Accuracy: {accuracy:.4f}")
    print(f"🔹 Precision: {precision:.4f}")
    print(f"🔹 Recall: {recall:.4f}")
    print(f"🔹 F1-score: {f1:.4f}")
    print(f"🔹 AUC: {auc:.4f}")
    print(f"\n🟢 Confusion Matrix:\n{cm}")
    print(f"\n📊 Classification Report:\n{classification_report(y_true, y_pred, digits=4)}")
    print(f"\n📂 Analysis Results saved in: {save_json_path}")

    metric = {
        "loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist()
    }

    result_str = f"Timestamp: {timestamp}\nMetric: {metric}\n\n"
    with open("result.txt", "a") as file:
        file.write(result_str)
    return metric, json_output

    

def add_arguments(parser):
    """

    """
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default='../data/benchmark.csv')
    parser.add_argument("--ckpt_output_path", type=str, default="../ckpts/finetune_ckpts")
    parser.add_argument("--model_output_path", type=str, default="../output")
    parser.add_argument("--result_save_path", type=str, default="../result")
    parser.add_argument("--latest_checkpoint", type=str, default="../ckpts/finetune_ckpts")
    parser.add_argument("--model_name", type=str, default="biot5")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    #args = task_dataset(args)
    print(args)
    #print(SUPPORTED_Tokenizer[args.model_name])
    #print(SUPPORTED_CKPT[args.model_name])
    tokenizer_org = SUPPORTED_Tokenizer[args.model_name].from_pretrained(SUPPORTED_CKPT[args.model_name])
    
    
    if(args.mode == 'encoder_check'):
        model = MainModel(args)
        print_model_info(model,level=2)
        model.language_model.main_model.resize_token_embeddings(len(tokenizer_org))
        #tokenizer_org = SUPPORTED_Tokenizer[args.model_name].from_pretrained(SUPPORTED_CKPT[args.model_name])

        test_dataset = MainDataset(split = "test",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        #print(test_dataset[1])
        print(f"dataset length {len(test_dataset)}")
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)
        for i, batch in enumerate(test_loader):
            #print(f"batch {i} : {batch}")   
            print(f"batch_loss {i} : {model(batch)}")
            if i >= 1:
                break
                
    if(args.mode == 'train'):
    best_losses = [None] * 5  # Store best loss for each fold
    fold_models = []  # Store trained models for each fold
    
    # Perform 5-fold cross-validation
    for fold in range(5):
        print(f"\nStarting Fold {fold+1}/5")
        logger.info(f"Starting Fold {fold+1}/5")
        
        # Model initialization for each fold
        logger.info("Loading model ......")
        model = MainModel(args)
        # Load checkpoint if available (e.g., from previous runs)
        latest_checkpoint = getattr(args, f'latest_checkpoint_fold_{fold}', None)
        if latest_checkpoint:
            print(f"Fold {fold+1} - Latest checkpoint: {latest_checkpoint}")
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_losses[fold] = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Fold {fold+1} - No checkpoint found.")
        print_model_info(model, level=2)
        logger.info("Loading model succeeded")
        
        # Dataset for the current fold
        logger.info(f"Loading dataset for Fold {fold+1} ......")
        train_dataset = MainDataset(
            split="train",
            fold=fold,
            tokenizer_org=tokenizer_org,
            args=args
        )
        valid_dataset = MainDataset(
            split="val",
            fold=fold,
            tokenizer_org=tokenizer_org,
            args=args
        )
        # Test dataset (same for all folds, not part of cross-validation)
        if fold == 0:  # Load test dataset only once
            test_dataset = MainDataset(
                split="test",
                fold=fold,  # fold parameter is ignored for test split
                tokenizer_org=tokenizer_org,
                args=args
            )
        
        logger.info("Loading dataset succeeded")

        # DataLoaders
        logger.info(f"Loading dataloader for Fold {fold+1} ......")
        train_loader = DataLoader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            args.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
        if fold == 0:  # Load test loader only once
            test_loader = DataLoader(
                test_dataset,
                args.batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn,
                pin_memory=True
            )
        logger.info("Loading dataloader succeeded")

        # Training setup
        model.to(args.device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        
        print(f"Now training Fold {fold+1}/5 on device: {args.device}")
        
        train_mol_decoder(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss)
        fold_models.append(model.state_dict())
        
    if(args.mode == 'infer'):
        args.latest_checkpoint = f"{args.latest_checkpoint}/{args.model_name}/6_20250416-1654.pth"
        #args.latest_checkpoint = None
        best_loss = None
        latest_checkpoint = args.latest_checkpoint

        model = MainModel(args)
        if latest_checkpoint is not None :
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict = True)
        print_model_info(model,level=2)
        logger.info("Loading model successed")
        
        # Molecules to predict
        molecules = [
            {"name": "Butonitazene", "smiles": "O=[N+](C1=CC=C2C(N=C(CC3=CC=C(OCCCC)C=C3)N2CCN(CC)CC)=C1)[O-]"},
            {"name": "3-Chloromethcathinone (3-CMC)", "smiles": "O=C(C1=CC(Cl)=CC=C1)C(C)NC"},
            {"name": "Dipentylone", "smiles": "CCCC(N(C)C)C(C1=CC=C(OCO2)C2=C1)=O"},
            {"name": "2-Fluorodeschloroketamine", "smiles": "O=C1C(NC)(C2=CC=CC=C2F)CCCC1"},
            {"name": "Bromazolam", "smiles": "CC1=NN=C2N1C3=CC=C(Br)C=C3C(C4=CC=CC=C4)=NC2 "}
        ]
        # Convert to canonical SMILES using RDKit
        # canonical_molecules = []
        # for mol in molecules:
        #     try:
        #         # Parse SMILES into RDKit molecule
        #         rdkit_mol = Chem.MolFromSmiles(mol["smiles"])
        #         if rdkit_mol is None:
        #             print(f"Error: Could not parse SMILES for {mol['name']}: {mol['smiles']}")
        #             continue
                
        #         # Convert to canonical SMILES
        #         canonical_smiles = Chem.MolToSmiles(rdkit_mol, canonical=True)
                
        #         # Store in the requested format
        #         canonical_molecules.append({
        #             "name": mol["name"],
        #             "smiles": canonical_smiles
        #         })
        #     except Exception as e:
        #         print(f"Error processing {mol['name']}: {e}")
        
        # # Assign to molecules.canonical
        # molecules = canonical_molecules
        
        # # Print results for verification
        # print("Canonical SMILES Results:")
        # for mol in molecules:
        #     print(f"Name: {mol['name']}, Canonical SMILES: {mol['smiles']}")
            
        # Initialize tokenizer and model
        tokenizer = tokenizer_org
        device = args.device
        model.to(device)
        
        # Predict labels for each molecule
        results = []
        for mol in molecules:
            # Tokenize the SMILES string
            tokenized_mol = tokenizer(
                mol["smiles"], return_tensors="pt", padding="max_length", truncation=True
            )
    
            # Move tensors to device
            tokenized_mol["input_ids"] = tokenized_mol["input_ids"].to(device)
            tokenized_mol["attention_mask"] = tokenized_mol["attention_mask"].to(device)
            
            # Predict using the model
            probability, predicted_class = model.predict(tokenized_mol)
            
            # Convert to CPU and extract values
            probability = probability.cpu().numpy()[0]  # Assuming batch_size=1
            predicted_class = predicted_class.cpu().numpy()[0]  # Assuming batch_size=1
            
            # Store results
            results.append({
                "Name": mol["name"],
                "SMILES": mol["smiles"],
                "Probability (NPS)": f"{probability:.4f}",
                "Predicted Label": "NPS" if predicted_class == 1 else "Non-NPS"
            })
        
        # Display results in a DataFrame
        results_df = pd.DataFrame(results)
        print("Prediction Results:")
        print(results_df.to_string(index=False))
            
    if(args.mode == 'model_check'):
        model = MainModel(args)
        print_model_info(model,level=2)
        
    if(args.mode == 'data_check'):
        #tokenizer_org = SUPPORTED_Tokenizer[args.model_name].from_pretrained(SUPPORTED_CKPT[args.model_name], padding_side='left')
        train_dataset = MainDataset(split = "train",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        test_dataset = MainDataset(split = "test",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        print(test_dataset[1])
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)
        
        for i, batch in enumerate(train_loader):
            print(f"Train Batch {i}: {batch}")
        
        for i, batch in enumerate(test_loader):
            if i >= 2:
                break
            else:
                print(f"Test Batch {i}: {batch}")

