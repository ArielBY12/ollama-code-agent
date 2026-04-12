"""JSON tool schemas sent to the LLM.

No imports, no side effects — pure data only.
Each schema follows the Ollama / OpenAI function-calling format.
"""

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. "
                "Always call this before editing a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or fully overwrite a file with the given content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": (
                "Replace one unique string in a file. "
                "Preferred for small, targeted edits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to patch.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": (
                            "Exact string to find (must be unique in file)."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Execute a shell command and return stdout, stderr, "
                "and exit code. Uses PowerShell on Windows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to run.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory path. Defaults to current dir."
                        ),
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": (
                            "If true, list recursively. Default false."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Regex code search across files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory to search in. Defaults to '.'."
                        ),
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": (
                            "Glob to filter files, e.g. '*.py'."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]
