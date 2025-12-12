"""
Starter scaffold for the CS 294-264 HW1 ReAct agent.

Students must implement a minimal ReAct agent that:
- Maintains a message history list (role, content, timestamp, unique_id)
- Uses a textual function-call format (see ResponseParser) with rfind-based parsing
- Alternates Reasoning and Acting until calling the tool `finish`
- Supports tools: `run_bash_cmd`, `finish`

This file intentionally omits core implementations and replaces them with
clear specifications and TODOs.
"""

from typing import List, Callable, Dict, Any
import time

from response_parser import ResponseParser
from llm import LLM, OpenAIModel
import inspect
from envs import LimitsExceeded

class ReactAgent:
    """
    Minimal ReAct agent that:
    - Maintains a message history list with unique ids
    - Builds the LLM context from the message list
    - Registers callable tools with auto-generated docstrings in the system prompt
    - Runs a Reason-Act loop until `finish` is called or MAX_STEPS is reached
    """

    def __init__(self, name: str, parser: ResponseParser, llm: LLM):
        self.name: str = name
        self.parser = parser
        self.llm = llm

        # Message list storage
        self.id_to_message: List[Dict[str, Any]] = []
        # Expose messages for trajectory dump utilities
        self.messages = self.id_to_message
        self.root_message_id: int = -1
        self.current_message_id: int = -1
        self._next_message_id: int = 0

        # Registered tools
        self.function_map: Dict[str, Callable] = {}

        # Set up the initial structure of the history
        # Create required root nodes and a user node (task)
        system_prompt = (
            "You are a careful ReAct agent. Think step-by-step, stay concise, and ALWAYS end with EXACTLY ONE function call.\n"
            "RESPONSE FORMAT (must always appear at end):\n"
            "your_thoughts_here\n"
            "...\n"
            "----BEGIN_FUNCTION_CALL----\n"
            "function_name\n"
            "----ARG----\n"
            "arg1_name\n"
            "----VALUE----\n"
            "arg1_value\n"
            "----ARG----\n"
            "arg2_name\n"
            "----VALUE----\n"
            "arg2_value\n"
            "...\n"
            "----END_FUNCTION_CALL----\n"
            "Rules: exactly one function call; exactly one END marker (do not repeat it); every argument must have both ARG and VALUE blocks; arguments MUST NOT contain BEGIN/END/ARG/VALUE markers; NEVER emit an empty function-call block—if you truly have nothing to do, call finish with an explanation.\n"
            "- Valid tools: finish(result: str) and all registered tools listed below.\n"
            "- Shell commands must be clean: DO NOT include BEGIN/END_FUNCTION_CALL or ARG/VALUE markers; keep them minimal.\n"
            "- After an error, fix inputs and retry rather than repeating the same failure.\n"
            "- Finish only once, and only after ensuring a meaningful result (prefer a small relevant test when code changed).\n"
            "- Be brief in reasoning; do not dump tool output (context is truncated)."
        )
        self.system_message_id = self.add_message("system", system_prompt)
        self.user_message_id = self.add_message("user", "")
        # NOTE: mandatory finish function that terminates the agent
        self.add_functions([self.finish])

    # -------------------- MESSAGE LIST --------------------
    def add_message(self, role: str, content: str) -> int:
        """
        Create a new message and add it to the list.

        The message must include fields: role, content, timestamp, unique_id.
        
        TODO(student): Implement this function to add a message to the list
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "unique_id": self._next_message_id,
        }
        self.id_to_message.append(message)
        self._next_message_id += 1
        return len(self.id_to_message) - 1

    def set_message_content(self, message_id: int, content: str) -> None:
        """
        Update message content by id.
        
        TODO(student): Implement this function to update a message's content
        """
        if not (0 <= message_id < len(self.id_to_message)):
            raise IndexError(f"Message id {message_id} is out of range.")
        self.id_to_message[message_id]["content"] = content

    def get_context(self, window: int = 30, include_system: bool = True) -> str:
        """
        Build the full LLM context from the message list.
        Uses a sliding window to avoid overlong prompts: always include the
        initial system and user messages, then the latest `window` messages.
        """
        total = len(self.id_to_message)
        if total <= 2:
            indices = range(total)
        else:
            tail_start = max(2, total - window)
            indices = [0, 1] + list(range(tail_start, total))

        if not include_system:
            indices = [idx for idx in indices if idx != self.system_message_id]

        contexts: List[str] = []
        for idx in indices:
            contexts.append(self.message_id_to_context(idx))
        return "\n".join(contexts)

    # -------------------- REQUIRED TOOLS --------------------
    def add_functions(self, tools: List[Callable]):
        """
        Add callable tools to the agent's function map.

        The system prompt must include tool descriptions that cover:
        - The signature of each tool
        - The docstring of each tool
        
        TODO(student): Implement this function to register tools and build tool descriptions
        """
        for tool in tools:
            if not callable(tool):
                raise ValueError(f"Tool {tool} is not callable.")
            self.function_map[tool.__name__] = tool
    
    def finish(self, result: str):
        """The agent must call this function with the final result when it has solved the given task. The function calls "git add -A and git diff --cached" to generate a patch and returns the patch as submission.

        Args: 
            result (str); the result generated by the agent

        Returns:
            The result passed as an argument.  The result is then returned by the agent's run method.
        """
        return result 

    # -------------------- MAIN LOOP --------------------
    def run(self, task: str, max_steps: int) -> str:
        """
        Run the agent's main ReAct loop:
        - Set the user prompt
        - Loop up to max_steps (<= 100):
            - Build context from the message list (with `message_id_to_context`)
            - Query the LLM
            - Parse a single function call at the end (see ResponseParser)
            - Execute the tool
            - Append tool result to the list
            - If `finish` is called, return the final result
            
        TODO(student): Implement the main ReAct loop
        """
        # Set the user task message
        # self.set_message_content(self.user_message_id, task)
        
        # Main ReAct loop
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        max_steps = min(max_steps, 100)

        self.set_message_content(self.user_message_id, task.strip())

        finish_attempts = 0

        for step in range(max_steps):
            system_context = self.message_id_to_context(self.system_message_id)
            conversation_context = self.get_context(include_system=False)
            llm_messages = [
                {"role": "system", "content": system_context},
                {"role": "user", "content": conversation_context},
            ]
            response = self.llm.generate(llm_messages)
            self.add_message("assistant", response)

            try:
                parsed = self.parser.parse(response)
            except ValueError as parse_error:
                debug_msg = (
                    f"Failed to parse LLM response: {parse_error}\n"
                    f"----RAW_RESPONSE_START----\n{response}\n----RAW_RESPONSE_END----"
                )
                print(debug_msg)
                self.add_message(
                    "system",
                    (
                        f"ParserError: {parse_error}. "
                        "You must ALWAYS end with a valid function call including function name and arguments, "
                        "following the specified template. Respond again using this exact format:\n"
                        f"{self.parser.response_format}"
                    ),
                )
                continue
            
            function_name = parsed["name"]
            arguments = self._sanitize_arguments(parsed["arguments"])
            tool = self.function_map.get(function_name)
            if tool is None:
                self.add_message(
                    "system",
                    f"Tool '{function_name}' is not registered. "
                    f"Valid tools: {', '.join(self.function_map.keys())}. "
                    "Retry with a valid tool and arguments.",
                )
                continue

            # Reject polluted arguments containing call markers for any tool
            polluted = False
            for val in arguments.values():
                if not isinstance(val, str):
                    continue
                if any(marker in val for marker in (self.parser.BEGIN_CALL, self.parser.END_CALL, self.parser.ARG_SEP, self.parser.VALUE_SEP)):
                    polluted = True
                    break
            if polluted:
                self.add_message(
                    "system",
                    "Argument values contained function-call markers (BEGIN/END/ARG/VALUE). Regenerate a clean function call with unpolluted arguments.",
                )
                continue

            try:
                tool_result = tool(**arguments)
            except Exception as e:
                tool_result = f"{type(e).__name__}: {e}"
                signature = inspect.signature(tool)
                self.add_message(
                    "system",
                    (
                        f"Tool '{function_name}{signature}' failed: {type(e).__name__}: {e}. "
                        "Retry with corrected arguments that match the signature."
                    ),
                )
            else:
                if not isinstance(tool_result, str):
                    tool_result = str(tool_result)

            # Record tool execution result
            tool_message_content = (
                f"{function_name}(args={arguments})\n"
                f"----RESULT----\n{tool_result}"
            )
            self.add_message("tool", tool_message_content)

            if tool is self.finish:
                finish_attempts += 1
                if not tool_result.strip():
                    self.add_message(
                        "system",
                        "finish returned an empty result. Ensure you generated a patch (git add -A && git diff --cached) or explain why no changes are needed, then call finish again.",
                    )
                    continue
                if finish_attempts > 1:
                    self.add_message(
                        "system",
                        "Duplicate finish detected earlier. Only call finish once with the final patch/result.",
                    )
                return tool_result

        raise LimitsExceeded(f"Reached {max_steps} steps without calling finish.")

    def _sanitize_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove any leaked function-call markers from string arguments by truncating
        at the first occurrence. This prevents commands like END_FUNCTION_CALL from
        being passed to tools.
        """
        cleaned = {}
        markers = (self.parser.BEGIN_CALL, self.parser.END_CALL, self.parser.ARG_SEP, self.parser.VALUE_SEP)
        for k, v in arguments.items():
            if isinstance(v, str):
                for marker in markers:
                    if marker in v:
                        v = v.split(marker, 1)[0].rstrip()
                        break
            cleaned[k] = v
        return cleaned

    def message_id_to_context(self, message_id: int) -> str:
        """
        Helper function to convert a message id to a context string.
        """
        message = self.id_to_message[message_id]
        header = f'----------------------------\n|MESSAGE(role="{message["role"]}", id={message["unique_id"]})|\n'
        content = message["content"]
        if message["role"] == "tool":
            # Show full output when the tool just ran (latest message);
            # compress only for older tool messages in history.
            if message_id != len(self.id_to_message) - 1:
                max_len = 2048
                if len(content) > max_len:
                    head = content[:1024]
                    tail = content[-512:]
                    content = f"{head}\n... [TRUNCATED {len(content) - (len(head)+len(tail))} CHARS] ...\n{tail}"
        if message["role"] == "system":
            tool_descriptions = []
            for tool in self.function_map.values():
                signature = inspect.signature(tool)
                docstring = inspect.getdoc(tool)
                tool_description = f"Function: {tool.__name__}{signature}\n{docstring}\n"
                tool_descriptions.append(tool_description)

            tool_descriptions = "\n".join(tool_descriptions)
            return (
                f"{header}{content}\n"
                f"--- AVAILABLE TOOLS ---\n{tool_descriptions}\n\n"
                f"--- RESPONSE FORMAT ---\n{self.parser.response_format}\n"
            )
        else:
            return f"{header}{content}\n"

def main():
    from envs import DumbEnvironment
    llm = OpenAIModel(None, "gpt-4o-mini")
    parser = ResponseParser()

    env = DumbEnvironment()
    dumb_agent = ReactAgent("dumb-agent", parser, llm)
    dumb_agent.add_functions([env.run_bash_cmd])
    result = dumb_agent.run("Show the contents of all files in the current directory.", max_steps=10)
    print(result)

if __name__ == "__main__":
    # Optional: students can add their own quick manual test here.
    main()
