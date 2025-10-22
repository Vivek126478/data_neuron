import sys
import os
sys.path.append(os.path.dirname(__file__))

from model import model_instance

def test_enhanced_similarity():
    """Test with more realistic text pairs"""
    
    test_cases = [
        {
            "text1": "The cat sat on the mat",
            "text2": "A cat was sitting on a mat",
            "expected": "high",
            "reason": "Same meaning, different words"
        },
        {
            "text1": "I love programming in Python",
            "text2": "Python programming is enjoyable", 
            "expected": "high",
            "reason": "Same topic, similar sentiment"
        },
        {
            "text1": "The weather is nice today",
            "text2": "It's sunny and warm outside",
            "expected": "high", 
            "reason": "Same concept, different phrasing"
        },
        {
            "text1": "Hello world",
            "text2": "Hello world", 
            "expected": "very high",
            "reason": "Identical texts"
        },
        {
            "text1": "The weather is nice today",
            "text2": "Technology companies are growing",
            "expected": "low",
            "reason": "Completely different topics"
        },
        {
            "text1": "Artificial intelligence is transforming technology",
            "text2": "AI is changing the tech industry", 
            "expected": "very high",
            "reason": "Synonyms and same meaning"
        },
        {
            "text1": "The company reported strong quarterly earnings",
            "text2": "Firm announces good financial results for the quarter",
            "expected": "high",
            "reason": "Business equivalent statements"
        }
    ]
    
    print("=== Enhanced MinHash Semantic Similarity Test ===")
    print()
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"--- Test Case {i+1} ---")
        print(f"Text1: '{test_case['text1']}'")
        print(f"Text2: '{test_case['text2']}'")
        print(f"Reason: {test_case['reason']}")
        print(f"Expected: {test_case['expected']}")
        
        score = model_instance.get_similarity_score(test_case['text1'], test_case['text2'])
        
        print(f"Similarity Score: {score}")
        
        # More nuanced validation
        if test_case['expected'] == 'very high':
            status = "✅ PASS" if score >= 0.8 else "❌ FAIL"
        elif test_case['expected'] == 'high':
            status = "✅ PASS" if score >= 0.5 else "⚠️  PARTIAL" if score >= 0.3 else "❌ FAIL"
        elif test_case['expected'] == 'low':
            status = "✅ PASS" if score < 0.3 else "❌ FAIL"
        
        print(f"{status}")
        print()
        
        results.append({
            'case': i+1,
            'expected': test_case['expected'],
            'score': score,
            'status': status
        })
    
    # Summary
    print("=== SUMMARY ===")
    passed = sum(1 for r in results if 'PASS' in r['status'])
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    test_enhanced_similarity()