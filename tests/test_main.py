import pytest
import csv
import os
import sys
from unittest.mock import patch, Mock, mock_open
sys.path.append('src')
from main import LLMEvaluator, run_evaluation, analyze_results

@pytest.fixture
def evaluator():
    return LLMEvaluator("test_api_key")

def test_evaluate_response_quality(evaluator):
    """Test response quality scoring."""
    # Good response
    good_response = """Artificial intelligence is a field of computer science that focuses on creating 
    systems capable of performing tasks that typically require human intelligence. For example, 
    machine learning algorithms can recognize patterns in data."""
    score = evaluator.evaluate_response_quality("What is AI?", good_response)
    assert 3 <= score <= 5
    
    # Poor response
    poor_response = "I don't know."
    score = evaluator.evaluate_response_quality("What is AI?", poor_response)
    assert score <= 2
    
    # Empty response
    score = evaluator.evaluate_response_quality("What is AI?", "")
    assert score == 1

def test_generate_response_success(evaluator):
    """Test successful response generation."""
    with patch('main.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "content": [{"text": "Hello! I'm an AI assistant."}]
        }
        mock_post.return_value = mock_response
        
        result = evaluator.generate_response("Hello", "You are helpful")
        
        assert result["success"] is True
        assert result["response"] == "Hello! I'm an AI assistant."
        assert result["latency_ms"] > 0
        assert result["error"] is None

def test_generate_response_failure(evaluator):
    """Test failed response generation."""
    with patch('main.requests.post') as mock_post:
        mock_post.side_effect = Exception("API Error")
        
        result = evaluator.generate_response("Hello", "You are helpful")
        
        assert result["success"] is False
        assert result["response"] is None
        assert result["error"] == "API Error"

def test_analyze_results():
    """Test results analysis."""
    # Create mock CSV data
    csv_data = """query,prompt_version,score,latency_ms,response,error
"What is AI?",A,4,1500,"AI is...",
"What is AI?",B,3,1200,"AI is cool! 😊",
"How to code?",A,5,1300,"To code...",
"How to code?",B,4,1400,"Coding is fun!",
"""
    
    with patch('builtins.open', mock_open(read_data=csv_data)):
        analysis = analyze_results("test.csv")
        
        assert analysis["prompt_A"]["count"] == 2
        assert analysis["prompt_B"]["count"] == 2
        assert analysis["prompt_A"]["mean_score"] == 4.5
        assert analysis["prompt_B"]["mean_score"] == 3.5
