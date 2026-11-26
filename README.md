# TeraBox data connector for Claude

Read-only remote MCP server. Claude (any claude.ai chat) queries your TeraBox files after
they were converted to Parquet in your private Hugging Face dataset by the Colab notebook.

Deploy free on Render (no card): open
`https://render.com/deploy?repo=https://github.com/<you>/terabox-connector`,
fill in `HF_TOKEN` (Hugging Face read token) and `HF_DATASET` (e.g. `you/terabox-data`).

Connector URL: `https://<service>.onrender.com/c/<ACCESS_KEY>/mcp`. Treat it like a password.
