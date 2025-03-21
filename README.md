# MCP Tool Langgraph Integration

## Description
Example project of how to integrate MCP endpoint tools into a Langgraph tool node

The graph consists of only 2 nodes, `agent` and `tool`.

## Prerequisites
To use this project, make sure you have Python 3.11.

### [uv](https://pypi.org/project/uv/) is recommended

#### Linux and Mac
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### MCP Server requirements  
- This example uses the MCP Server sample `@modelcontextprotocol/server-brave-search` to add Brave Search tools. This requires that you have `node` and `npx` installed.

### API Keys
- The MCP Server sample used is for Brave Search, you can get a free API key from https://brave.com/search/api/
- You will need and API key for the chosen AI provider which defaults to Anthropic but can be changed by editing the `__main__.py` file
- Put all api keys in a .env file in the repository root.

## create a config.json in the root of the repo

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-brave-search"
      ],
      "env": {
        "BRAVE_API_KEY": "YOUR BRAVE API KEY"
      }
    }
  }
}

```

## create a .env file in the root fo the repo with your anthropic api key
```
ANTHROPIC_API_KEY=YOUR_KEY
```

## From source Usage
```shell
uv run mcp_langgraph_tools "your question that needs search"
```

## Whats New

- Version 0.2.0:
  - Initial release

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Paul Robello - probello@gmail.com
