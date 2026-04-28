import os
import json
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
import openai


class JarvisAI:
    """Jarvis AI Personal Assistant with conversational capabilities using OpenAI GPT-4o."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Jarvis AI with OpenAI API key.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass it as an argument to JarvisAI()."
            )
        self.client = openai.OpenAI(api_key=self.api_key)
        self.conversation_history: List[Dict[str, str]] = []
        self.model = "gpt-4o"
        self.name = "Jarvis"
        self._load_system_prompt()

    def _load_system_prompt(self) -> None:
        """Load the system prompt defining Jarvis's personality and capabilities."""
        self.system_prompt = (
            "You are Jarvis, a helpful and intelligent AI personal assistant. "
            "Your goal is to assist users with their questions, tasks, and daily needs "
            "in a friendly, professional, and efficient manner.\n\n"
            "Personality traits:\n"
            "- Polite, respectful, and helpful\n"
            "- Concise but thorough in your responses\n"
            "- Proactive: offer suggestions when appropriate\n"
            "- Capable of handling a wide range of tasks and topics\n\n"
            "Capabilities:\n"
            "- Answer questions on various topics\n"
            "- Help with writing, editing, and brainstorming\n"
            "- Assist with planning and organization\n"
            "- Explain complex concepts clearly\n"
            "- Engage in thoughtful conversation\n\n"
            "Current time context: Always be aware that you're running as an AI assistant "
            "and your knowledge has a cutoff. For time-sensitive information, note that "
            "you may not have the most current data."
        )

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """Build the message list for the OpenAI API call.

        Args:
            user_input: The user's latest message.

        Returns:
            List of message dicts for the conversation.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})
        return messages

    async def chat_async(
        self,
        message: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Send a message to Jarvis and get a response asynchronously.

        Args:
            message: The user's message.
            temperature: Sampling temperature (0.0 to 2.0). Lower = more deterministic.
            max_tokens: Maximum tokens to generate. None uses API default.
            stream: If True, returns a streaming response iterator.

        Returns:
            Dict with 'response', 'usage', and 'timestamp' keys.
            If stream=True, returns the streaming iterator directly.
        """
        if stream:
            return self._stream_response(message, temperature, max_tokens)

        messages = self._build_messages(message)

        try:
            response = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )

            assistant_reply = response.choices[0].message.content
            usage = response.usage

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": assistant_reply})

            return {
                "response": assistant_reply,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except openai.OpenAIError as e:
            return {
                "response": f"Error communicating with OpenAI: {str(e)}",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "timestamp": datetime.now().isoformat(),
                "error": True,
            }

    def _stream_response(
        self, message: str, temperature: float, max_tokens: Optional[int]
    ):
        """Generate a streaming response from the OpenAI API.

        Args:
            message: The user's message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            A generator yielding chunks of text.
        """
        messages = self._build_messages(message)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        full_response = ""
        try:
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
        finally:
            if full_response:
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": full_response})

    def chat(self, message: str, **kwargs) -> Dict[str, Any]:
        """Send a message to Jarvis and get a response (synchronous wrapper).

        Args:
            message: The user's message.
            **kwargs: Passed to chat_async.

        Returns:
            Dict with response, usage, and timestamp.
        """
        return asyncio.run(self.chat_async(message, **kwargs))

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the full conversation history.

        Returns:
            List of conversation message dicts.
        """
        return self.conversation_history.copy()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()

    def update_system_prompt(self, prompt: str) -> None:
        """Update the system prompt (Jarvis's personality/instructions).

        Args:
            prompt: New system prompt string.
        """
        self.system_prompt = prompt

    def set_model(self, model: str) -> None:
        """Set the OpenAI model to use.

        Args:
            model: Model name (e.g., 'gpt-4o', 'gpt-4o-mini').
        """
        self.model = model


def interactive_mode():
    """Run Jarvis in interactive command-line mode."""
    print("=" * 60)
    print(" Jarvis AI Personal Assistant")
    print("=" * 60)
    print("Powered by OpenAI GPT-4o")
    print("Type 'quit', 'exit', or 'bye' to end the session.")
    print("Type 'history' to view conversation history.")
    print("Type 'clear' to clear history.")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  Warning: OPENAI_API_KEY environment variable not set.")
        api_key = input("Enter your OpenAI API key: ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    try:
        jarvis = JarvisAI()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("Please set the OPENAI_API_KEY environment variable and try again.")
        return

    print(f"\n✅ Jarvis is ready! (Model: {jarvis.model})\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            lower_input = user_input.lower()
            if lower_input in ("quit", "exit", "bye"):
                print("Jarvis: Goodbye! It was nice chatting with you.")
                break

            if lower_input == "history":
                history = jarvis.get_conversation_history()
                if history:
                    print("\n--- Conversation History ---")
                    for i, msg in enumerate(history, 1):
                        role = msg["role"].capitalize()
                        content = msg["content"]
                        print(f"\n{i}. [{role}]: {content[:200]}")
                        if len(content) > 200:
                            print("   ...")
                    print()
                else:
                    print("No conversation history yet.\n")
                continue

            if lower_input == "clear":
                jarvis.clear_history()
                print("Conversation history cleared.\n")
                continue

            print("Jarvis: ", end="", flush=True)

            result = jarvis.chat(user_input, temperature=0.7)

            if result.get("error"):
                print(f"\n{result['response']}\n")
            else:
                print(f"\n{result['response']}\n")
                usage = result.get("usage", {})
                if usage:
                    print(f"[Tokens: {usage.get('prompt_tokens', 0)} prompt + "
                          f"{usage.get('completion_tokens', 0)} completion = "
                          f"{usage.get('total_tokens', 0)} total]")
                print()

        except KeyboardInterrupt:
            print("\n\nJarvis: Goodbye! It was nice chatting with you.")
            break
        except EOFError:
            print("\n\nJarvis: Goodbye! It was nice chatting with you.")
            break


if __name__ == "__main__":
    interactive_mode()
