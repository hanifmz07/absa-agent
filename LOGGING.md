# Logging System Guide

All `print()` statements have been replaced with a proper logging system using Python's built-in `logging` module.

## Quick Start

To enable logging in your scripts, add this at the beginning of your main script:

```python
from logging_config import setup_logging
import logging

# Configure logging
setup_logging(level=logging.INFO, log_file="app.log")
```

## Logging Levels

The logging system uses different levels to categorize messages:

- **DEBUG**: Detailed diagnostic information (e.g., processing individual cases)
- **INFO**: General informational messages (e.g., model initialization, completion messages)
- **WARNING**: Warning messages for issues that don't prevent execution (e.g., retries, forced outputs)
- **ERROR**: Error messages for serious problems (e.g., max retries reached)

## Examples

### Basic Usage (Console Only)

```python
from logging_config import setup_logging
import logging

# Log to console only
setup_logging(level=logging.INFO)

# Your code here...
```

### Advanced Usage (Console + File)

```python
from logging_config import setup_logging
import logging

# Log to both console and file
setup_logging(level=logging.DEBUG, log_file="logs/absa_agent.log")

# Your code here...
```

### Changing Log Level

```python
# For production - show only INFO and above
setup_logging(level=logging.INFO)

# For debugging - show all messages including DEBUG
setup_logging(level=logging.DEBUG)

# For quiet mode - show only warnings and errors
setup_logging(level=logging.WARNING)
```

## Modified Files

The following files have been updated to use logging instead of print:

- `src/utils/agent.py` - ABSASystem (synchronous version)
- `src/utils/agent_async.py` - AsyncABSASystem (asynchronous version)
- `src/main/run_agent.py` - Main runner script (sync)
- `src/main/run_agent_async.py` - Main runner script (async)
- `src/main/eval.py` - Evaluation script for causal LM
- `src/main/eval_seq2seq.py` - Evaluation script for seq2seq models

## Using Logging in New Code

When adding new code, use the logging module instead of print:

```python
import logging

# At the top of your module
logger = logging.getLogger(__name__)

# In your functions
logger.debug("Detailed information for debugging")
logger.info("General informational message")
logger.warning("Warning message")
logger.error("Error message")
```

## Integration with Existing Scripts

To use logging in your existing scripts:

1. Import and configure logging at the start of your script:
   ```python
   from logging_config import setup_logging
   import logging
   
   if __name__ == "__main__":
       setup_logging(level=logging.INFO, log_file="my_script.log")
       # Rest of your code...
   ```

2. The classes and functions in `src/` will automatically use the configured logging.

## Output Format

The default log format is:
```
2026-02-23 10:30:45 - module_name - INFO - Your log message here
```

This includes:
- Timestamp
- Module name
- Log level
- Message

## Notes

- Third-party library logs (transformers, vllm, torch) are set to WARNING level by default to reduce noise
- All log messages are now properly categorized by severity
- Log files are appended to (not overwritten) by default
