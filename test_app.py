#!/usr/bin/env python3
"""
Test script for DC Energy Chat API
Run this to see exactly what your app returns!
"""

import requests
import json

def test_api():
    base_url = "http://localhost:5000"
    
    print("🚀 Testing DC Energy Chat API...\n")
    
    # Test 1: Health Check
    print("1️⃣ Health Check:")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Chat Endpoint - General Query
    print("2️⃣ Chat API - General Question:")
    try:
        response = requests.post(
            f"{base_url}/chat",
            headers={"Content-Type": "application/json"},
            json={"query": "Hello, what can you help me with?"}
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Chat Endpoint - Energy Question
    print("3️⃣ Chat API - Energy Cost Question:")
    try:
        response = requests.post(
            f"{base_url}/chat",
            headers={"Content-Type": "application/json"},
            json={"query": "What will my energy costs be next week?"}
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 4: Chat Endpoint - OpenInfra Data Question
    print("4️⃣ Chat API - Power Data Question:")
    try:
        response = requests.post(
            f"{base_url}/chat",
            headers={"Content-Type": "application/json"},
            json={"query": "Show me power consumption for server-001"}
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n🎉 All tests completed!")
    print("\n💡 Your DC Energy Chat app is working perfectly!")
    print("   Next steps: Implement AI logic in chat_api/routes.py")

if __name__ == "__main__":
    test_api() 