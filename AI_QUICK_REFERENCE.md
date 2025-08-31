# AI Coding Assistant Quick Reference

## GitHub Copilot Shortcuts

| Action | Shortcut | Description |
|--------|----------|-------------|
| Accept suggestion | `Tab` | Accept the current suggestion |
| Reject suggestion | `Esc` | Dismiss the current suggestion |
| Next suggestion | `Alt + ]` | Show next suggestion |
| Previous suggestion | `Alt + [` | Show previous suggestion |
| Trigger suggestion | `Ctrl + Space` | Manually trigger suggestions |
| Inline chat | `Ctrl + I` | Open inline chat |
| Chat panel | `Ctrl + Shift + I` | Open chat in side panel |

## Useful Prompts for Deep Learning

### Neural Network Components
```python
# Create a dense layer with weights and biases initialization
# Implement forward propagation for a neural network layer
# Add dropout layer for regularization
# Create activation function with derivative
```

### Training & Optimization
```python
# Implement gradient descent optimizer
# Add early stopping mechanism
# Create learning rate scheduler
# Implement batch normalization
```

### Data Processing
```python
# Normalize image data for neural network
# Create data loader with batch processing
# Implement train/validation split
# Add data augmentation for images
```

### Metrics & Logging
```python
# Calculate classification accuracy
# Implement confusion matrix visualization
# Log metrics to Wandb
# Create training progress bar
```

## Chat Commands

| Command | Usage | Example |
|---------|-------|---------|
| `/explain` | Explain code | `/explain how does backpropagation work?` |
| `/fix` | Fix bugs | `/fix this matrix dimension error` |
| `/generate` | Generate code | `/generate a ReLU activation function` |
| `/optimize` | Optimize code | `/optimize this training loop for speed` |
| `/test` | Create tests | `/test create unit tests for this function` |

## Best Practices

1. **Write descriptive comments** before code
2. **Use meaningful variable names**
3. **Break complex functions into smaller ones**
4. **Always review AI-generated code**
5. **Test thoroughly before using**

## Troubleshooting AI Suggestions

- **No suggestions?** Check internet connection and Copilot status
- **Poor suggestions?** Add more context with comments
- **Wrong language?** Check file extension and language mode
- **Slow responses?** Try restarting VS Code

## Project-Specific Tips

For this deep learning assignment:
- Comment your mathematical formulas clearly
- Describe the shape of tensors in comments
- Explain the purpose of each layer
- Document hyperparameter choices
- Use descriptive names for loss functions and optimizers