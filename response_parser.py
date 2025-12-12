class ResponseParser:
    """
    Parses LLM responses to extract a single function call using a rigid textual format.

    The LLM must output exactly one function call at the end of its response.
    Do NOT use JSON or XML. Use rfind to locate the final markers.
    """

    BEGIN_CALL = "----BEGIN_FUNCTION_CALL----"
    END_CALL = "----END_FUNCTION_CALL----"
    ARG_SEP = "----ARG----"
    VALUE_SEP = "----VALUE----"

    # Students should include this exact template in the system prompt so the LLM follows it.
    response_format = f"""
your_thoughts_here
...
{BEGIN_CALL}
function_name
{ARG_SEP}
arg1_name
{VALUE_SEP}
arg1_value (can be multiline)
{ARG_SEP}
arg2_name
{VALUE_SEP}
arg2_value (can be multiline)
...
{END_CALL}

Rules:
- Exactly one function call per response.
- function_name MUST match a registered tool (e.g., run_bash_cmd, finish).
- Every argument must include both a name and value block, even if the value is empty.
- Never omit the final function call.

DO NOT CHANGE ANY TEST! AS THEY WILL BE USED FOR EVALUATION.
"""

    def parse(self, text: str) -> dict:
        """
        Parse the function call from `text` using string.rfind to avoid confusion with
        earlier delimiter-like content in the reasoning.

        Returns a dictionary: {"thought": str, "name": str, "arguments": dict}
        
        TODO(student): Implement this function using rfind to parse the function call
        """
        if not isinstance(text, str):
            raise TypeError("LLM response must be a string.")

        end_idx = text.rfind(self.END_CALL)
        if end_idx == -1:
            raise ValueError("Missing END_FUNCTION_CALL marker.")

        begin_idx = text.rfind(self.BEGIN_CALL, 0, end_idx)
        if begin_idx == -1:
            raise ValueError("Missing BEGIN_FUNCTION_CALL marker.")

        thought = text[:begin_idx].strip()
        call_block = text[begin_idx + len(self.BEGIN_CALL):end_idx].strip()
        if not call_block:
            raise ValueError("Empty function call block.")
        else:
            print(f"Call block: {call_block}")

        sections = call_block.split(self.ARG_SEP)
        function_name = sections[0].strip()
        if not function_name:
            raise ValueError("Function name is missing.")
        else:
            print(f"Function name: {function_name}")

        arguments = {}
        for section in sections[1:]:
            if not section.strip():
                continue
            if self.VALUE_SEP not in section:
                raise ValueError("Argument missing VALUE separator.")
            arg_name_part, value_part = section.split(self.VALUE_SEP, 1)
            arg_name = arg_name_part.strip()
            if not arg_name:
                raise ValueError("Argument name is empty.")
            arg_value = value_part.strip()
            arguments[arg_name] = arg_value
            print(f"Argument: {arg_name} = {arg_value}")

        print("--------------------------------")
        return {"thought": thought, "name": function_name, "arguments": arguments}
