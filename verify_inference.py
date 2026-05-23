import sys
import os
import json

# Add azure_deployment to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'azure_deployment'))

import score

def test_local_scoring():
    print("Initializing scoring service...")
    score.init()
    
    # Test cases:
    # 1. Warm start user (e.g. user_id = 112)
    # 2. Cold start user (e.g. user_id = 9999999)
    test_cases = [
        {"user_id": 3107, "top_k": 5, "threshold": 0.0},
        {"user_id": 9999999, "top_k": 5, "threshold": 0.0}
    ]
    
    for case in test_cases:
        print(f"\nTesting request payload: {case}")
        raw_data = json.dumps(case)
        response = score.run(raw_data)
        
        # Load and pretty print response
        try:
            res_dict = json.loads(response)
            print(f"Mode: {res_dict.get('mode')}")
            if "error" in res_dict:
                print(f"Error: {res_dict['error']}")
            else:
                print("Predictions:")
                for idx, pred in enumerate(res_dict.get("predictions", [])):
                    print(f"  {idx+1}. Product ID: {pred['product_id']}, Name: {pred['product_name']}, Probability: {pred['reorder_probability']:.4f}")
        except Exception as e:
            print(f"Raw Response parsing error: {e}")
            print(f"Raw Response: {response}")

if __name__ == "__main__":
    test_local_scoring()
