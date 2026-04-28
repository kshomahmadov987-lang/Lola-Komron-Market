# Jarvis AI Personal Assistant

A conversational AI personal assistant powered by OpenAI's GPT-4o model.

## Features

- **Conversational Capabilities**: Natural, context-aware conversations with memory of previous interactions
- **GPT-4o Integration**: Leverages OpenAI's latest GPT-4o model for intelligent responses
- **Async Support**: Built-in asynchronous API for non-blocking operations
- **Streaming Responses**: Real-time streaming of AI responses
- **Conversation History**: Automatic tracking of conversation context
- **Customizable Personality**: Configurable system prompts and behavior
- **Token Usage Tracking**: Monitor API usage with detailed token counts

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### As a Module

```python
import asyncio
from main import JarvisAI

# Initialize
jarvis = JarvisAI()

# Send a message
result = jarvis.chat("Hello, how are you?")
print(result['response'])

# Async usage
async def chat():
    result = await jarvis.chat_async("Tell me about GPT-4o")
    print(result['response'])
    print(f"Tokens used: {result['usage']['total_tokens']}")

asyncio.run(chat())

# Streaming response
async def stream_chat():
    async for chunk in jarvis.chat_async("Write a story", stream=True):
        print(chunk, end="", flush=True)

asyncio.run(stream_chat())
```

### Interactive Mode

Run the assistant interactively:
```bash
python main.py
```

Commands in interactive mode:
- Type your message and press Enter to chat
- `quit`, `exit`, or `bye` - End the session
- `history` - View conversation history
- `clear` - Clear conversation history
- Ctrl+C or Ctrl+D - Exit

## API Reference

### JarvisAI Class

#### `__init__(api_key: Optional[str] = None)`
Initialize the assistant with an OpenAI API key.

**Args:**
- `api_key`: OpenAI API key (falls back to OPENAI_API_KEY env var)

#### `chat_async(message: str, temperature: float = 0.7, max_tokens: Optional[int] = None, stream: bool = False)`
Send a message asynchronously.

**Args:**
- `message`: User's message
- `temperature`: Sampling temperature (0.0-2.0)
- `max_tokens`: Maximum tokens to generate
- `stream`: Return streaming iterator if True

**Returns:**
Dict with `response`, `usage`, and `timestamp` keys (or iterator if streaming)

#### `chat(message: str, **kwargs)`
Synchronous wrapper for `chat_async`.

#### `get_conversation_history()`
Get a copy of the conversation history.

#### `clear_history()`
Clear all conversation history.

#### `update_system_prompt(prompt: str)`
Update Jarvis's system instructions/personality.

#### `set_model(model: str)`
Change the OpenAI model (default: 'gpt-4o').

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Model Options

- `gpt-4o` (default): Latest and most capable model
- `gpt-4o-mini`: Faster, cheaper alternative
- `gpt-4`: Previous generation model

## Examples

### Basic Chat

```python
from main import JarvisAI

jarvis = JarvisAI()
result = jarvis.chat("Help me write a Python function to sort a list")
print(result['response'])
```

### Multiple Messages with Context

```python
jarvis = JarvisAI()
jarvis.chat("I'm planning a trip to Japan")
jarvis.chat("What are some must-visit cities?")
# Jarvis remembers the context about the Japan trip
```

### Custom System Prompt

```python
jarvis = JarvisAI()
jarvis.update_system_prompt(
    "You are a coding expert. Always provide Python code examples."
)
result = jarvis.chat("How do I read a CSV file?")
```

## Requirements

- Python 3.7+
- openai>=1.0.0

## License

MIT
