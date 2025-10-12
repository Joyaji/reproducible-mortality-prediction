# Contributing to In-Hospital Mortality Prediction

Thank you for your interest in contributing! This project aims to provide a reproducible, open-source implementation of deep learning for mortality prediction.

## 🎯 How to Contribute

### **1. Reporting Issues**

If you find a bug or have a suggestion:

1. **Check existing issues** to avoid duplicates
2. **Open a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Relevant logs or screenshots

### **2. Proposing Enhancements**

For new features or improvements:

1. **Open an issue first** to discuss the proposal
2. Explain:
   - What problem it solves
   - How it aligns with project goals
   - Potential implementation approach
3. Wait for feedback before starting work

### **3. Submitting Code**

#### **Setup Development Environment**

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/reproducible-mortality-prediction.git
cd reproducible-mortality-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

#### **Development Workflow**

1. **Create a branch:**


   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes:**

   - Write clean, documented code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation

3. **Test your changes:**

   ```bash
   # Run tests
   pytest tests/

   # Check code style
   black src/ --check
   flake8 src/

   # Type checking
   mypy src/
   ```

4. **Commit your changes:**

   ```bash
   git add .
   git commit -m "feat: add feature description"
   # or
   git commit -m "fix: fix bug description"
   ```

5. **Push and create Pull Request:**

   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a PR on GitHub with a clear description.

## 📝 Code Style Guidelines

### **Python Code**

- **Formatting:** Use `black` (line length: 100)
- **Linting:** Follow `flake8` rules
- **Type hints:** Add type annotations for functions
- **Docstrings:** Use Google-style docstrings

Example:

```python
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50
) -> keras.Model:
    """
    Train LSTM model with Focal Loss.
    
    Args:
        X_train: Training features (n_samples, timesteps, features)
        y_train: Training labels (n_samples,)
        epochs: Number of training epochs
        
    Returns:
        Trained Keras model
        
    Raises:
        ValueError: If input shapes are invalid
    """
    # Implementation
    pass
```

### **Documentation**

- **Markdown:** Use proper headings, lists, and code blocks
- **Comments:** Explain *why*, not *what*
- **README:** Update if adding new features
- **REPRODUCIBILITY.md:** Update if changing workflow

### **Commit Messages**

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add attention mechanism to LSTM
fix: correct calibration curve calculation
docs: update reproduction steps in README
refactor: simplify data loading pipeline
```

## 🧪 Testing

### **Running Tests**

```bash
# All tests
pytest

# Specific test file
pytest tests/test_data_loader.py

# With coverage
pytest --cov=src tests/
```

### **Writing Tests**

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Test edge cases and error conditions

Example:

```python
def test_data_loader_handles_missing_values():
    """Test that DataLoader correctly handles missing values."""
    loader = MIMICDataLoader('data/test')
    X, y = loader.load_data('test')
    
    assert not np.isnan(X).any(), "Data should not contain NaN"
    assert X.shape[0] == y.shape[0], "X and y must have same length"
```

## 📊 Adding New Features

### **Synthetic Data Generator**

If modifying `src/generate_synthetic_data.py`:

- Maintain realistic clinical patterns
- Document new parameters
- Update `docs/SYNTHETIC_DATA_GENERATOR.md`

### **Model Architecture**

If modifying `src/train_dl.py`:

- Ensure backward compatibility
- Document hyperparameters
- Update `docs/DEEP_LEARNING_MODEL.md`
- Regenerate results and update `TECHNICAL_REPORT.md`

### **Visualizations**

If adding plots to `src/generate_plots.py`:

- Follow existing style (300 DPI, consistent colors)
- Add to `src/generate_report.py`
- Update documentation

## 🚫 What NOT to Commit

- ❌ Large model files (use Git LFS or releases)
- ❌ Full datasets (provide generation scripts)
- ❌ Personal credentials or API keys
- ❌ IDE-specific files (already in `.gitignore`)
- ❌ Temporary or log files

## 📚 Resources

- **Original Paper:** Rajkomar et al. (2018) - [Link](https://doi.org/10.1038/s41746-018-0029-1)
- **MIMIC-III:** [PhysioNet](https://physionet.org/content/mimiciii/)
- **Project Documentation:** See `docs/` directory
- **Reproduction Guide:** `REPRODUCIBILITY.md`

## 🤝 Code of Conduct

This project follows a Code of Conduct to ensure a welcoming environment:

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

See `CODE_OF_CONDUCT.md` for details.

## ❓ Questions?

- **General questions:** Open a GitHub Discussion
- **Bug reports:** Open an Issue
- **Security issues:** Email maintainers directly (see README)

## 🙏 Recognition

Contributors will be acknowledged in:

- GitHub contributors list
- Project documentation
- Future publications (for significant contributions)

---

Thank you for contributing to open science and reproducible research! 🚀
