import sys
import os
import unittest

from edges_lang_chain_impl import NodesLangChainImpl

class TestNodes(unittest.TestCase):
    def setUp(self):
        self.nodes = NodesLangChainImpl("llama3.2:3b-instruct-fp16")
    
    def test_retrieve(self):
        self.assertEqual(self.nodes.retrieve("What is the chemical equilibrium?"), [])
    
    def test_router(self):
        self.assertEqual(self.nodes.router("What is the chemical equilibrium?"), [])
        self.assertEqual(self.nodes.router("What are the models released today for llama3.2?"), []) 
        self.assertEqual(self.nodes.router("What are the types of agent memory?"), [])
        self.assertEqual(self.nodes.router("Who is favored to win the NFC Championship game in the 2024 season?"), [])
    
    def test_grade_relevance(self):
        self.assertEqual(self.nodes.grade_relevance(["What is the chemical equilibrium?"], "What is the chemical equilibrium?"), "no")

if __name__ == '__main__':
    unittest.main()
