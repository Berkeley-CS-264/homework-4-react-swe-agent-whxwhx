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
        # Registered tools
        self.function_map: Dict[str, Callable] = {}

        # Set up the initial structure of the history
        # Create required root nodes and a user node (task)
        system_prompt = (
            "You are a careful ReAct-style software engineering agent operating inside a real codebase.\n"
            "\n"
            "Your job: solve the user's task by iterating between (1) brief reasoning and (2) tool use.\n"
            "You have access to tools described below; use them to inspect files, run commands, and produce a final patch/result.\n"
            "\n"
            "CRITICAL OUTPUT RULES (must follow exactly):\n"
            "- Every assistant message MUST end with EXACTLY ONE function call using the provided textual protocol.\n"
            "- The function call block MUST be the final thing in the message (no extra text after END marker).\n"
            "- Do NOT output JSON or XML. Do NOT output multiple function calls.\n"
            "- Do NOT include any of these markers anywhere except in the final call block: "
            "----BEGIN_FUNCTION_CALL----, ----END_FUNCTION_CALL----, ----ARG----, ----VALUE----.\n"
            "- Argument values may be multiline.\n"
            "\n"
            "HOW TO WORK:\n"
            "- Prefer small, verifiable steps. When unsure, inspect the repo using run_bash_cmd.\n"
            "- If a tool fails, read the error and retry with corrected arguments.\n"
            "- Before you finish, run a simple test to verify the changes you made.\n"
            "\n"
            "FINISHING:\n"
            "- Only call finish when the task is complete.\n"
            "- When you complete, call finish(result=...) where you explain the change in a few sentences.\n"
            "- Never call finish with an empty result.\n"
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
        self.current_message_id += 1
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "unique_id": self.current_message_id,
        }
        self.id_to_message.append(message)
        return self.current_message_id

    def set_message_content(self, message_id: int, content: str) -> None:
        """
        Update message content by id.
        
        TODO(student): Implement this function to update a message's content
        """
        if not (0 <= message_id <= self.current_message_id):
            raise IndexError(f"Message id {message_id} is out of range.")
        self.id_to_message[message_id]["content"] = content

    def get_context(self, include_system: bool = True) -> str:
        """
        Build the full LLM context from the message list.
        
        TODO(student): Implement this function to build the context from the message list
        """
        context = ""
        for i in range(self.current_message_id + 1):
            if not include_system and i == self.system_message_id:
                continue
            context += self.message_id_to_context(i) + "\n"
        return context

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

        for step in range(max_steps):
            system_context = self.message_id_to_context(self.system_message_id)
            conversation_context = self.get_context(include_system=False)
            llm_messages = [
                {'role': self.id_to_message[i]['role'], 'content': self.message_id_to_context(i)}
                for i in range(self.current_message_id + 1)
            ] + [{'role': 'system', 'content': self.message_id_to_context(self.system_message_id)}]
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
            arguments = parsed["arguments"]
            tool = self.function_map.get(function_name)
            if tool is None:
                self.add_message(
                    "system",
                    f"Tool '{function_name}' is not registered. "
                    f"Valid tools: {', '.join(self.function_map.keys())}. "
                    "Retry with a valid tool and arguments.",
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
                f"----TOOL RESULT BEGIN----\n{tool_result}\n----TOOL RESULT END----\n"
            )
            self.add_message("user", tool_message_content)

            if "finish" in function_name:
                return tool_result  

        raise LimitsExceeded(f"Reached {max_steps} steps without calling finish.")

    def message_id_to_context(self, message_id: int) -> str:
        """
        Helper function to convert a message id to a context string.
        """
        message = self.id_to_message[message_id]
        header = (
            f'----------------------------\n'
            f'|MESSAGE(role="{message["role"]}", id={message["unique_id"]})|\n'
        )
        content = message["content"]
        if message["role"] == "tool":
            # Show full output when the tool just ran (latest message);
            # compress only for older tool messages in history.
            if message_id != len(self.id_to_message) - 1:
                max_len = 4096
                if len(content) > max_len:
                    head = content[:2048]
                    tail = content[-2048:]
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
                f"--- AVAILABLE TOOLS ---\n{tool_descriptions}\n----AVAILABLE TOOLS END----\n"
                f"--- RESPONSE FORMAT ---\n{self.parser.response_format}\n----RESPONSE FORMAT END----\n"
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
