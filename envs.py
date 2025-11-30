from utils import get_sb_environment
import subprocess
import swebench
import base64
import shlex

class LimitsExceeded(Exception):
    """Raised when the agent has reached its step limit."""


class SWEEnvironment:
    """
    Minimal interface to the SWEBench execution environment.

    Students may use their own wrapper. The environment must expose:
    - execute(command: str) -> str: Run a shell command and return stdout, or raise ValueError on failure
    """

    def __init__(self, instance: dict):
        self.env = get_sb_environment(instance)
        self.instance = instance  # Store instance for test execution
     
    # -------------------- REQUIRED TOOLS --------------------
    def run_bash_cmd(self, command: str) -> str:
        """
        Run the command in a bash shell and return the output or throw a ValueError
        if the process returns non-zero exit code.

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        try:
            output = self.env.execute(command)
            
            # Handle case where execute returns a dict instead of string
            if isinstance(output, dict):
                output = output.get("output", "") or output.get("stdout", "")
                
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ValueError(output)
        except TimeoutError:
            raise ValueError("TimeoutError")
        return output
    
    def generate_patch(self, result: str) -> str:
        """
        Generate a patch from the result (for SWE-Bench)
        """
        try:
            patch_output = self.env.execute("git add -A && git diff --cached")
            
            # Handle case where execute returns a dict instead of string
            if isinstance(patch_output, dict):
                patch_output = patch_output.get("output", "") or patch_output.get("stdout", "")
            
            if patch_output and patch_output.strip():
                return patch_output
            else:
                return f"{result}\n\nNo changes detected to generate a patch."
        except Exception as e:
            return f"{result}\n\nError running git commands: {e}"

    # -------------------- TODO(student): add more functions here if you want, not required --------------------
    def replace_in_file(self, file_path: str, from_line: int, to_line: int, content: str) -> str:
        """
        [Optional] Replace the content of the file from the given line to the given line with the given content
        """
        if from_line < 1 or to_line < from_line:
            raise ValueError("Invalid line range supplied.")

        encoded_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        command = f"""python - <<'PY'
import base64
from pathlib import Path

path = Path(r"{file_path}")
if not path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

text = path.read_text()
lines = text.splitlines(keepends=True)

start = max(0, {from_line} - 1)
end = min(len(lines), max(start, {to_line}))

replacement_text = base64.b64decode("{encoded_content}").decode("utf-8")
replacement_lines = replacement_text.splitlines(keepends=True)

new_lines = lines[:start] + replacement_lines + lines[end:]
new_text = "".join(new_lines)

path.write_text(new_text)
print(f"Replaced lines {from_line}-{to_line} in {file_path}")
PY"""
        return self.env.execute(command)

    def show_file(self, file_path: str) -> str:
        """
        [Optional]Show the content of the file
        """
        quoted_path = shlex.quote(file_path)
        return self.env.execute(f"cat {quoted_path}")

class DumbEnvironment:
    """
    Dumb environment that just executes the command
    """

    def execute(self, command: str) -> str:
        """
        Run the command in bash and return the output

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        result = subprocess.run(command, capture_output=True, shell=True, check=False)
        output = f"--STDOUT--\n{result.stdout.decode()}\n--STDERR--\n{result.stderr.decode()}"
        if result.returncode:
            raise ValueError(output)
        return output
    
    def run_bash_cmd(self, command: str) -> str:
        """
        Run the command in a bash shell and return the output or throw a ValueError
        if the process returns non-zero exit code.

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        return self.execute(command)
