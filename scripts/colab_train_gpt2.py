#!/usr/bin/env python3
"""
Quick GPT-2 fine-tuning for Colab.
"""

import os
import sys
import json
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from colab_helper import mount_drive, setup_path, sync_paths_to_drive, write_json


def train_gpt2(max_lines: int = 50000, max_steps: int = 1000, drive_dir: str | None = None):
    """Fine-tune GPT-2 on Wikitext-103 within a bounded Colab budget."""
    
    print("=" * 60)
    print("GPT-2 FINE-TUNING")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install: pip install transformers datasets")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using: {device}")
    
    # Load model & tokenizer
    print("\n🤖 Loading GPT-2...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Fix: GPT-2 doesn't have pad token by default
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    model.to(device)
    
    # Load training data
    print("\n📚 Loading data...")
    train_file = 'data/wikitext-103/wiki.train.tokens'
    
    # Read and prepare data
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    lines = [l.strip() for l in lines if l.strip()][:max_lines]
    
    print(f"   Loaded {len(lines):,} lines")
    
    # Tokenize
    print("   Tokenizing...")
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=256, padding='max_length')
    
    dataset = Dataset.from_dict({'text': lines})
    train_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    print(f"   Train: {len(train_dataset):,} samples")
    
    # Setup training
    print("\n🎯 Starting training...")
    
    output_dir = 'artifacts/gpt2_finetuned'
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=3e-5,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=1,
        max_steps=max_steps,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    start_time = time.time()
    trainer.train()
    elapsed = (time.time() - start_time) / 60
    
    print(f"\n✅ Training complete in {elapsed:.1f} minutes")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model')
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"💾 Model saved to: {final_model_path}")
    
    train_logs = [entry for entry in trainer.state.log_history if "loss" in entry]
    final_loss = train_logs[-1]["loss"] if train_logs else None
    rough_ppl = torch.exp(torch.tensor(final_loss)).item() if final_loss is not None else None
    
    # Save results
    results = {
        'train_time_minutes': elapsed,
        'final_train_loss': final_loss,
        'rough_train_ppl': rough_ppl,
        'steps_trained': trainer.state.global_step,
        'max_lines': max_lines,
        'max_steps': max_steps,
    }
    
    write_json(os.path.join(output_dir, 'training_results.json'), results)
    
    print(f"\n💾 Results saved to: {output_dir}/training_results.json")

    if drive_dir:
        backup_root = mount_drive(drive_dir)
        if backup_root:
            sync_paths_to_drive(["artifacts/gpt2_finetuned"], backup_root, project_root)
    
    return final_model_path


if __name__ == "__main__":
    import argparse

    setup_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lines", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--drive-dir", default="DA_NPL_Artifacts")
    parser.add_argument("--skip-drive-sync", action="store_true")
    args = parser.parse_args()
    train_gpt2(
        max_lines=args.max_lines,
        max_steps=args.max_steps,
        drive_dir=None if args.skip_drive_sync else args.drive_dir,
    )
