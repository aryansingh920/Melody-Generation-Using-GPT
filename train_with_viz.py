"""
Created on 23/12/2024

@author: Aryan

Filename: train_with_viz.py

Relative Path: train_with_viz.py
"""

import torch
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import wandb

from gpt_downsizing import create_custom_config, create_custom_config_1, create_custom_config_2, create_custom_config_3
from gpt import GPTLanguageModel, generate_melody, decode, estimate_loss, get_batch
from MelodyDataset import ModelConfig


def save_model_and_metrics(model, config, training_history, metrics, name, save_dir='saved_models'):
    """Save model weights, config, and training metrics."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, f'{name}_model.pt')
    torch.save(model.state_dict(), model_path)

    metrics_path = os.path.join(save_dir, f'{name}_metrics.json')
    config_dict = {
        k: str(v) if isinstance(v, torch.device) else v
        for k, v in config.__dict__.items()
    }

    metrics_data = {
        'config': config_dict,
        'training_history': {
            'train_losses': [float(x) for x in training_history['train_losses']],
            'val_losses': [float(x) for x in training_history['val_losses']]
        },
        'final_metrics': metrics
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Model and metrics saved to {save_dir}/{name}_*")


def test_configs_with_tracking():
    """
    Trains multiple configurations (Original, Deeper_Thinner, Wider_Shallower)
    and tracks them with wandb. Returns the best model based on final val loss.
    """
    configs = [
        ("Original", create_custom_config()),
        ("Deeper_Thinner", create_custom_config_1()),
        ("Wider_Shallower", create_custom_config_2()),
        # Uncomment if you want to test the bigger config
        # ("Custom_Config_3", create_custom_config_3())
    ]

    all_results = []
    plt.figure(figsize=(15, 10))

    for name, config in configs:
        print(f"\nTesting {name} Configuration:")
        for key, value in config.__dict__.items():
            print(f"{key}: {value}")

        # Init wandb
        wandb.init(project="gpt-melody-generation", name=name,
                   config=config.__dict__, reinit=True)

        # Create the model with the config
        model = GPTLanguageModel(
            config, vocab_size_override=None).to(config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate)

        training_history = {'train_losses': [], 'val_losses': []}

        pbar = tqdm(range(config.max_iters), desc=f"Training {name}")
        for iter in pbar:
            # Evaluate losses at intervals
            if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
                losses = estimate_loss()
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                training_history['train_losses'].append(losses['train'])
                training_history['val_losses'].append(losses['val'])

                wandb.log({
                    "train_loss": losses['train'],
                    "val_loss": losses['val'],
                    "step": iter
                })

                pbar.set_postfix({
                    'train_loss': f"{losses['train']:.4f}",
                    'val_loss': f"{losses['val']:.4f}"
                })

            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        steps = list(range(0, config.max_iters + 1, config.eval_interval))
        # Just in case the final step isn't exactly at max_iters, slice the arrays properly
        plt.plot(steps[:len(training_history['train_losses'])],
                 training_history['train_losses'], label=f'{name} (Train)', linestyle='-')
        plt.plot(steps[:len(training_history['val_losses'])],
                 training_history['val_losses'], label=f'{name} (Val)', linestyle='--')

        # Generate sample melody
        prompt = "R F G A "
        generated_melody = generate_melody(prompt, max_new_tokens=200)
        print(f"\nGenerated melody for {name}:\n{generated_melody}")

        final_train_loss = training_history['train_losses'][-1]
        final_val_loss = training_history['val_losses'][-1]
        overfitting_metric = final_val_loss - final_train_loss

        metrics = {
            'final_train_loss': float(final_train_loss),
            'final_val_loss': float(final_val_loss),
            'overfitting_metric': float(overfitting_metric),
            'params': sum(p.numel() for p in model.parameters()),
            'generated_sample': generated_melody
        }

        # Log final metrics
        wandb.log(metrics)

        # Save model and metrics
        save_model_and_metrics(model, config, training_history, metrics, name)

        result = {
            "name": name,
            "config": config,
            "model": model,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "overfitting_metric": overfitting_metric,
            "generated_text": generated_melody,
            "training_history": training_history
        }
        all_results.append(result)
        wandb.finish()

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/loss_comparison.png')
    plt.close()

    print("\nPerformance Comparison:")
    print("=" * 80)
    print(f"{'Config':<20} {'Final Train Loss':<15} {'Final Val Loss':<15} {'Overfitting':<15}")
    print("-" * 80)
    for result in all_results:
        print(f"{result['name']:<20} {result['final_train_loss']:<15.4f} {result['final_val_loss']:<15.4f} {result['overfitting_metric']:<15.4f}")

    create_final_comparison_plot(all_results)

    # Find best model based on val loss
    best_model_idx = min(range(len(all_results)),
                         key=lambda i: all_results[i]['final_val_loss'])
    best_result = all_results[best_model_idx]
    print(
        f"\nBest model: {best_result['name']} with validation loss: {best_result['final_val_loss']:.4f}")

    return best_result["model"]


def create_final_comparison_plot(results):
    plt.figure(figsize=(10, 5))
    names = [r['name'] for r in results]
    train_losses = [r['final_train_loss'] for r in results]
    val_losses = [r['final_val_loss'] for r in results]

    x = range(len(names))
    width = 0.35

    plt.bar([i - width/2 for i in x], train_losses,
            width, label='Train Loss', color='skyblue')
    plt.bar([i + width/2 for i in x], val_losses, width,
            label='Validation Loss', color='lightcoral')

    plt.xlabel('Configuration')
    plt.ylabel('Final Loss')
    plt.title('Final Training and Validation Loss Comparison')
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/final_loss_comparison.png')
    plt.close()


if __name__ == "__main__":
    print("Starting configuration comparison...")
    best_model = test_configs_with_tracking()
