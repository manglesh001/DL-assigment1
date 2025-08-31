# VS Code AI Agents Setup Guide

This guide will help you set up AI coding assistance (agents) in your local VS Code environment for this Deep Learning assignment project.

## Prerequisites

1. **VS Code**: Download and install [Visual Studio Code](https://code.visualstudio.com/)
2. **Python 3.8+**: Ensure Python is installed on your system
3. **Git**: Ensure Git is installed for version control

## Step 1: Clone and Open the Repository

```bash
git clone https://github.com/manglesh001/DL-assigment1.git
cd DL-assigment1
code .
```

## Step 2: Install Recommended Extensions

When you open the project in VS Code, you'll see a notification to install recommended extensions. Click "Install All" or install them manually:

### Essential AI Coding Assistants:
- **GitHub Copilot** (`github.copilot`) - Primary AI coding assistant
- **GitHub Copilot Chat** (`github.copilot-chat`) - AI chat integration
- **GitHub Copilot Labs** (`github.copilot-labs`) - Experimental AI features
- **TabNine** (`tabnine.tabnine-vscode`) - Alternative AI completion
- **IntelliCode** (`visualstudioexptteam.vscodeintellicode`) - Microsoft AI assistance

### Python Development:
- **Python** (`ms-python.python`) - Core Python support
- **Pylance** (`ms-python.vscode-pylance`) - Advanced Python language server
- **Jupyter** (`ms-toolsai.jupyter`) - Jupyter notebook support
- **Black Formatter** (`ms-python.black-formatter`) - Code formatting
- **Flake8** (`ms-python.flake8`) - Linting

### Additional Productivity:
- **GitLens** (`eamodio.gitlens`) - Enhanced Git capabilities
- **Remote Development** extensions for containers/SSH/WSL

## Step 3: Set Up Python Environment

1. **Create Virtual Environment:**
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure Environment Variables:**
```bash
cp .env.example .env
# Edit .env file with your Wandb credentials
```

## Step 4: Enable AI Features

### GitHub Copilot Setup:
1. **Get GitHub Copilot Access:**
   - Sign up for [GitHub Copilot](https://github.com/features/copilot)
   - Students can get it free through GitHub Student Pack

2. **Sign in to GitHub in VS Code:**
   - Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
   - Run "GitHub: Sign in"
   - Complete authentication

3. **Configure Copilot:**
   - Open Command Palette
   - Run "GitHub Copilot: Sign In"
   - Enable inline suggestions

### TabNine Setup (Alternative):
1. Install TabNine extension
2. Sign up for TabNine Pro (optional but recommended for AI features)
3. Configure in settings

## Step 5: Configure AI-Friendly Settings

The project includes optimized settings in `.vscode/settings.json` that:
- Enable AI completions and suggestions
- Configure Python analysis for better AI context
- Set up formatting and linting
- Optimize Jupyter notebook integration

## Step 6: Using AI Agents

### GitHub Copilot Usage:
- **Inline Suggestions**: Start typing code, Copilot suggests completions
- **Chat**: Use `Ctrl+I` to open inline chat for code explanations
- **Generate**: Describe what you want in comments, Copilot generates code
- **Explain**: Select code and ask Copilot to explain it

### Example Prompts for Your DL Project:
```python
# Generate a function to implement ReLU activation
# Create a neural network layer with forward propagation
# Implement Adam optimizer for gradient descent
# Add logging for training metrics to Wandb
```

### Copilot Chat Commands:
- `/explain` - Explain selected code
- `/fix` - Fix bugs in code
- `/generate` - Generate code from description
- `/optimize` - Optimize code performance
- `/test` - Generate unit tests

## Step 7: Debugging and Development

### Launch Configurations:
Use the pre-configured debug settings in `.vscode/launch.json`:
- **Train Script**: Debug the main training script
- **Current File**: Debug any Python file
- **Custom Args**: Train with custom parameters

### Tasks:
Use built-in tasks (Ctrl+Shift+P → "Tasks: Run Task"):
- **Install Dependencies**: Install required packages
- **Run Training**: Execute training script
- **Format Code**: Auto-format with Black
- **Lint Code**: Check code quality
- **Run Jupyter**: Start Jupyter server

## Step 8: AI-Assisted Development Workflow

1. **Start with Comments**: Write descriptive comments about what you want to implement
2. **Let AI Generate**: Allow Copilot to suggest implementations
3. **Review and Refine**: Always review AI-generated code
4. **Use Chat for Help**: Ask questions about deep learning concepts
5. **Debug with AI**: Use AI to explain errors and suggest fixes

## Troubleshooting

### Copilot Not Working:
1. Check internet connection
2. Verify GitHub Copilot subscription
3. Sign out and sign in again
4. Restart VS Code

### Python Environment Issues:
1. Ensure correct Python interpreter is selected
2. Check virtual environment activation
3. Verify all dependencies are installed

### Jupyter Issues:
1. Install ipykernel: `pip install ipykernel`
2. Register kernel: `python -m ipykernel install --user --name=venv`
3. Select correct kernel in Jupyter notebooks

## Tips for Better AI Assistance

1. **Write Clear Comments**: AI works better with clear context
2. **Use Descriptive Variable Names**: Helps AI understand intent
3. **Break Down Complex Tasks**: Split large functions into smaller ones
4. **Ask Specific Questions**: Be specific in chat queries
5. **Verify AI Output**: Always test and verify generated code

## Project-Specific AI Prompts

For this deep learning assignment, try these prompts:

```python
# Implement forward propagation for a multi-layer neural network
# Create a function to compute cross-entropy loss with regularization  
# Generate code to visualize training loss curves with matplotlib
# Implement data preprocessing for Fashion-MNIST dataset
# Create hyperparameter sweep configuration for Wandb
# Add early stopping mechanism to prevent overfitting
```

## Getting Help

- **GitHub Copilot Docs**: [https://docs.github.com/en/copilot](https://docs.github.com/en/copilot)
- **VS Code Python**: [https://code.visualstudio.com/docs/python/python-tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- **Jupyter in VS Code**: [https://code.visualstudio.com/docs/datascience/jupyter-notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

## Next Steps

1. Open a Python file or Jupyter notebook
2. Start coding with AI assistance
3. Use the debugging configurations to test your code
4. Experiment with different AI prompts to learn deep learning concepts
5. Use Wandb integration to track your experiments

Happy coding with AI assistance! 🚀