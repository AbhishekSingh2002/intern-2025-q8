# Q8: Offline Evaluation A/B Testing

A/B testing framework that evaluates two different system prompts across 20 diverse queries, measuring quality scores and latency.

## Features

- **Two System Prompts**: Professional vs Enthusiastic styles
- **20 Diverse Queries**: Covering various topics and complexity levels  
- **Automated Scoring**: Heuristic-based quality evaluation (1-5 scale)
- **Performance Metrics**: Response latency measurement
- **CSV Output**: Structured results for analysis
- **Visualization**: Matplotlib plots comparing prompts
- **Statistical Analysis**: Mean scores, distributions, best/worst queries

## System Prompts

- **Prompt A (Professional)**: Formal, concise, clear responses
- **Prompt B (Enthusiastic)**: Creative, engaging, emoji-friendly responses

## Scoring Criteria

Quality scores (1-5) based on:
- Response length appropriateness
- Structure and organization
- Use of examples and explanations
- Topic engagement and relevance
- Completeness and helpfulness

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Mistral API key:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

## Usage

1. Run the evaluation:
   ```bash
   python src/main.py
   ```
   This will generate `evaluation_results.csv` with all the evaluation data.

2. Generate analysis and plots:
   ```bash
   python analysis.py
   ```

   > **Note**: If you're using the provided Mistral API key, please be aware that it might have rate limits. For production use, use your own API key.
   This will create `evaluation_results.png` with visualizations and print a detailed analysis.

## Project Structure

```
intern-2025-q8/
├── src/
│   └── main.py           # Main evaluation script
├── tests/
│   └── test_main.py      # Unit tests
├── analysis.py           # Analysis and visualization
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Running Tests

```bash
pytest tests/ -v
```

## Example Output

```
=== LLM A/B Testing Evaluation ===
Testing 20 queries with 2 system prompts
This will take several minutes...
--------------------------------------------------

=== Analysis Results ===
Prompt A (Professional):
  Mean Score: 4.2/5
  Mean Latency: 1450ms

Prompt B (Enthusiastic):
  Mean Score: 3.8/5
  Mean Latency: 1380ms

Winners:
  Quality: Prompt A
  Speed: Prompt B
```

## Results Visualization

![Evaluation Results](evaluation_results.png)

## Notes

- The evaluation may take some time to complete as it makes API calls to the Mistral API.
- Results are saved to `evaluation_results.csv` for further analysis.
- The analysis script generates visualizations in `evaluation_results.png`.
