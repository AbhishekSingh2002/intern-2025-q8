import os
import csv
import time
import json
from typing import Dict, List, Any
from dotenv import load_dotenv
import requests
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# System prompts for A/B testing
SYSTEM_PROMPTS = {
    "A": """You are a helpful, professional AI assistant. Provide clear, accurate, and concise responses to user questions. Be informative and maintain a formal but friendly tone.""",
    
    "B": """You are an enthusiastic and creative AI assistant! Use engaging language, emojis where appropriate, and provide detailed explanations with examples. Make your responses fun and memorable while staying accurate."""
}

# Test queries covering diverse topics and complexities
TEST_QUERIES = [
    "What is artificial intelligence?",
    "How do I bake a chocolate cake?",
    "Explain quantum computing in simple terms",
    "What are the benefits of renewable energy?",
    "How do I write a professional email?",
    "What is the difference between Python and Java?",
    "Tell me about the history of the internet",
    "How can I improve my productivity?",
    "What is climate change and why is it important?",
    "Explain how machine learning works",
    "What are some good books for learning programming?",
    "How do I start a small business?",
    "What is blockchain technology?",
    "How can I learn a new language effectively?",
    "What are the health benefits of exercise?",
    "Explain the concept of supply and demand",
    "How do computers process information?",
    "What is the scientific method?",
    "How can I manage stress better?",
    "What are some tips for public speaking?"
]

class LLMEvaluator:
    """
    Evaluates LLM responses using different system prompts.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def generate_response(self, query: str, system_prompt: str) -> Dict[str, Any]:
        """
        Generate response using specified system prompt.
        
        Args:
            query: User query
            system_prompt: System prompt to use
            
        Returns:
            Dict with response text and metadata
        """
        data = {
            "model": "mistral-tiny",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=data, timeout=30)
            end_time = time.time()
            
            response.raise_for_status()
            result = response.json()
            
            response_text = result["choices"][0]["message"]["content"].strip()
            latency_ms = round((end_time - start_time) * 1000, 2)
            
            logger.info(
                "Response generated",
                query_length=len(query),
                response_length=len(response_text),
                latency_ms=latency_ms
            )
            
            return {
                "response": response_text,
                "latency_ms": latency_ms,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            end_time = time.time()
            latency_ms = round((end_time - start_time) * 1000, 2)
            
            logger.error("Response generation failed", error=str(e), latency_ms=latency_ms)
            
            return {
                "response": None,
                "latency_ms": latency_ms,
                "success": False,
                "error": str(e)
            }
    
    def evaluate_response_quality(self, query: str, response: str) -> int:
        """
        Simple heuristic-based scoring of response quality.
        
        Args:
            query: Original user query
            response: Generated response
            
        Returns:
            Score from 1-5
        """
        if not response or response.strip() == "":
            return 1
        
        score = 3  # Base score
        
        # Length appropriateness (not too short, not too long)
        if len(response) < 50:
            score -= 1  # Too short
        elif len(response) > 1000:
            score -= 0.5  # Possibly too long
        elif 100 <= len(response) <= 500:
            score += 0.5  # Good length
        
        # Check for structure and completeness
        if any(marker in response.lower() for marker in ["first", "second", "finally", "in conclusion"]):
            score += 0.5  # Well-structured
        
        # Check for examples or explanations
        if any(word in response.lower() for word in ["example", "for instance", "such as", "like"]):
            score += 0.5  # Includes examples
        
        # Check for question engagement
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        if overlap >= 2:
            score += 0.5  # Good topic engagement
        
        # Penalize generic or unhelpful responses
        if any(phrase in response.lower() for phrase in ["i can't", "i don't know", "sorry, i cannot"]):
            score -= 1
        
        # Bonus for comprehensive responses
        if len(response) > 200 and any(word in response.lower() for word in ["because", "since", "therefore", "however"]):
            score += 0.5  # Explains reasoning
        
        # Ensure score is in valid range
        return max(1, min(5, round(score)))

def run_evaluation(api_key: str, output_file: str = "evaluation_results.csv"):
    """
    Run A/B testing evaluation on test queries.
    
    Args:
        api_key: Anthropic API key
        output_file: CSV file to save results
    """
    evaluator = LLMEvaluator(api_key)
    results = []
    
    logger.info("Starting evaluation", total_queries=len(TEST_QUERIES))
    
    for i, query in enumerate(TEST_QUERIES, 1):
        logger.info(f"Evaluating query {i}/{len(TEST_QUERIES)}", query=query)
        
        for prompt_version in ["A", "B"]:
            system_prompt = SYSTEM_PROMPTS[prompt_version]
            
            logger.info("Generating response", prompt_version=prompt_version)
            result = evaluator.generate_response(query, system_prompt)
            
            if result["success"]:
                score = evaluator.evaluate_response_quality(query, result["response"])
                logger.info("Response scored", prompt_version=prompt_version, score=score)
            else:
                score = 1  # Failed responses get minimum score
                logger.warning("Response failed", prompt_version=prompt_version)
            
            # Store result
            results.append({
                "query": query,
                "prompt_version": prompt_version,
                "score": score,
                "latency_ms": result["latency_ms"],
                "response": result["response"] if result["success"] else None,
                "error": result["error"]
            })
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
    
    # Save results to CSV
    logger.info("Saving results", output_file=output_file, total_results=len(results))
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["query", "prompt_version", "score", "latency_ms", "response", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    logger.info("Evaluation completed", output_file=output_file)
    return results

def analyze_results(csv_file: str = "evaluation_results.csv") -> Dict[str, Any]:
    """
    Analyze evaluation results and compute statistics.
    
    Args:
        csv_file: CSV file with results
        
    Returns:
        Analysis statistics
    """
    results_a = []
    results_b = []
    
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["prompt_version"] == "A":
                results_a.append({
                    "score": int(row["score"]),
                    "latency_ms": float(row["latency_ms"])
                })
            else:
                results_b.append({
                    "score": int(row["score"]),
                    "latency_ms": float(row["latency_ms"])
                })
    
    # Calculate statistics
    def calc_stats(results):
        if not results:
            return {"mean_score": 0, "mean_latency": 0, "count": 0}
        
        scores = [r["score"] for r in results]
        latencies = [r["latency_ms"] for r in results]
        
        return {
            "mean_score": round(sum(scores) / len(scores), 2),
            "mean_latency": round(sum(latencies) / len(latencies), 2),
            "count": len(results)
        }
    
    stats_a = calc_stats(results_a)
    stats_b = calc_stats(results_b)
    
    analysis = {
        "prompt_A": stats_a,
        "prompt_B": stats_b,
        "winner": {
            "score": "A" if stats_a["mean_score"] > stats_b["mean_score"] else "B",
            "latency": "A" if stats_a["mean_latency"] < stats_b["mean_latency"] else "B"
        }
    }
    
    logger.info("Analysis completed", analysis=analysis)
    return analysis

def main():
    load_dotenv()
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")
    
    # Run evaluation
    print("=== LLM A/B Testing Evaluation ===")
    print(f"Testing {len(TEST_QUERIES)} queries with 2 system prompts")
    print("This will take several minutes...")
    print("-" * 50)
    
    results = run_evaluation(api_key)
    
    # Analyze results
    print("\n=== Analysis Results ===")
    analysis = analyze_results()
    
    print(f"Prompt A (Professional):")
    print(f"  Mean Score: {analysis['prompt_A']['mean_score']}/5")
    print(f"  Mean Latency: {analysis['prompt_A']['mean_latency']}ms")
    
    print(f"Prompt B (Enthusiastic):")
    print(f"  Mean Score: {analysis['prompt_B']['mean_score']}/5")
    print(f"  Mean Latency: {analysis['prompt_B']['mean_latency']}ms")
    
    print(f"\nWinners:")
    print(f"  Quality: Prompt {analysis['winner']['score']}")
    print(f"  Speed: Prompt {analysis['winner']['latency']}")
    
    print(f"\nResults saved to: evaluation_results.csv")
    print("Run 'python analysis.py' to generate plots!")
    
    return 0

if __name__ == "__main__":
    exit(main())
