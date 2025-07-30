#!/usr/bin/env python3
"""
Synthetic Dataset Generator for LLM Inference Testing

This script generates realistic synthetic datasets for testing LLM inference
with specified number of prompts, input lengths, and output lengths.
"""

import random
import json
import argparse
from typing import List, Tuple, Optional
from transformers import PreTrainedTokenizerBase, AutoTokenizer


class SyntheticDatasetGenerator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        
        # Sample corpus for realistic text generation
        self.corpus = """
        The advancement of artificial intelligence has transformed numerous industries and continues to reshape our world.
        Machine learning algorithms process vast amounts of data to identify patterns and make predictions.
        Natural language processing enables computers to understand and generate human language with increasing sophistication.
        Deep learning architectures have revolutionized computer vision, speech recognition, and text generation.
        Climate change represents one of the most pressing challenges facing humanity in the 21st century.
        Renewable energy sources like solar and wind power are becoming increasingly cost-effective and efficient.
        Quantum computing promises to solve complex problems that are intractable for classical computers.
        Blockchain technology has applications beyond cryptocurrency, including supply chain management and digital identity.
        Biotechnology advances are enabling personalized medicine and novel therapeutic approaches.
        Space exploration continues to expand our understanding of the universe and our place within it.
        Cybersecurity measures must evolve to address increasingly sophisticated threats in our digital world.
        Data privacy and ethical AI development are crucial considerations for technology companies and policymakers.
        The intersection of technology and healthcare is producing innovative solutions for diagnosis and treatment.
        Educational technology is transforming how we learn and acquire new skills throughout our lives.
        Sustainable development requires balancing economic growth with environmental protection and social equity.
        """ * 10  # Repeat to ensure sufficient content
        
        self.words = self.corpus.split()
        
        # Prompt templates for different tasks
        self.templates = {
            'analysis': [
                "Analyze the following topic in detail: {topic}. Consider the implications, challenges, and potential solutions.",
                "Provide a comprehensive evaluation of {topic}, including its advantages, disadvantages, and future prospects.",
                "Examine the relationship between {topic1} and {topic2}, discussing their interconnections and mutual influence.",
                "Assess the impact of {topic} on society, economy, and technology over the next decade."
            ],
            'explanation': [
                "Explain the concept of {concept} to {audience}. Include relevant examples and practical applications.",
                "Describe how {process} works, breaking down the key steps and underlying principles.",
                "Compare and contrast {item1} and {item2}, highlighting their similarities and differences.",
                "Define {term} and discuss its significance in the context of {domain}."
            ],
            'creative': [
                "Write a creative piece about {scenario}. Make it engaging and thought-provoking.",
                "Develop a story that explores the theme of {theme} in a futuristic setting.",
                "Create a dialogue between two experts discussing {topic} from different perspectives.",
                "Compose a detailed description of {setting} that captures both its beauty and complexity."
            ],
            'instruction': [
                "Provide step-by-step instructions for {task}. Include safety considerations and best practices.",
                "Create a comprehensive guide for {activity}, suitable for beginners but thorough enough for reference.",
                "Outline the process of {procedure}, explaining the rationale behind each step.",
                "Develop a tutorial for {skill}, including common mistakes to avoid and tips for success."
            ],
            'problem_solving': [
                "How would you solve the following problem: {problem}? Provide multiple approaches and evaluate their effectiveness.",
                "Address the challenge of {challenge} by proposing innovative solutions and implementation strategies.",
                "What are the best practices for handling {situation}? Include preventive measures and response protocols.",
                "Analyze this scenario: {scenario}. What recommendations would you make for improvement?"
            ]
        }
        
        # Content pools for template filling
        self.content_pools = {
            'topics': [
                'artificial intelligence', 'climate change', 'quantum computing', 'blockchain technology',
                'renewable energy', 'space exploration', 'biotechnology', 'cybersecurity', 'data privacy',
                'machine learning', 'neural networks', 'autonomous vehicles', 'smart cities', 'IoT devices',
                'sustainable development', 'digital transformation', 'remote work', 'virtual reality',
                'gene therapy', 'precision medicine', '5G networks', 'edge computing', 'cloud infrastructure'
            ],
            'concepts': [
                'deep learning', 'natural language processing', 'computer vision', 'reinforcement learning',
                'distributed systems', 'microservices architecture', 'containerization', 'DevOps practices',
                'agile methodology', 'data structures', 'algorithms', 'database optimization', 'API design',
                'user experience', 'accessibility', 'performance optimization', 'scalability patterns'
            ],
            'audiences': [
                'beginners', 'intermediate learners', 'advanced practitioners', 'business stakeholders',
                'technical teams', 'students', 'researchers', 'industry professionals', 'policymakers',
                'general public', 'subject matter experts', 'cross-functional teams'
            ],
            'processes': [
                'machine learning model training', 'software development lifecycle', 'data pipeline construction',
                'system architecture design', 'security audit procedures', 'performance testing protocols',
                'user research methodologies', 'agile sprint planning', 'continuous integration deployment'
            ],
            'domains': [
                'healthcare', 'finance', 'education', 'manufacturing', 'retail', 'transportation',
                'energy', 'agriculture', 'entertainment', 'telecommunications', 'government', 'research'
            ],
            'scenarios': [
                'a world where AI has solved climate change', 'the first human colony on Mars',
                'a society with universal basic income', 'cities powered entirely by renewable energy',
                'a future where quantum computers are commonplace', 'global collaboration on pandemic response'
            ],
            'themes': [
                'human-AI collaboration', 'ethical technology development', 'sustainable innovation',
                'digital equity', 'privacy in the digital age', 'the future of work', 'environmental stewardship'
            ]
        }
    
    def pad_to_length(self, text: str, target_len: int) -> str:
        """Pad or truncate text to achieve target token length."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        current_len = len(tokens)
        
        if current_len >= target_len:
            return self.tokenizer.decode(tokens[:target_len])
        
        # Pad with realistic content
        padding_phrases = [
            " Furthermore, it's important to consider the broader implications of this topic.",
            " Additionally, we should examine the various factors that contribute to this phenomenon.",
            " Moreover, the long-term effects of these developments cannot be overlooked.",
            " In this context, we can observe significant patterns emerging across different sectors.",
            " It's worth noting that recent research has provided new insights into this area.",
            " From this perspective, we can identify several key trends that merit attention.",
            " Building on this foundation, we can explore additional aspects of the subject.",
            " Taking into account current developments, several conclusions can be drawn.",
            " Given the complexity of this issue, multiple approaches may be necessary.",
            " Considering the rapid pace of change, continuous adaptation is essential."
        ]
        
        while current_len < target_len:
            padding = random.choice(padding_phrases)
            new_tokens = self.tokenizer.encode(padding, add_special_tokens=False)
            tokens.extend(new_tokens)
            current_len = len(tokens)
            
            # Add some randomness to avoid exact repetition
            if current_len < target_len - 50:
                random_words = random.sample(self.words, min(10, target_len - current_len))
                random_text = " " + " ".join(random_words)
                random_tokens = self.tokenizer.encode(random_text, add_special_tokens=False)
                tokens.extend(random_tokens)
                current_len = len(tokens)
        
        return self.tokenizer.decode(tokens[:target_len])
    
    def generate_template_based_prompt(self) -> str:
        """Generate a prompt using templates and content pools."""
        category = random.choice(list(self.templates.keys()))
        template = random.choice(self.templates[category])
        
        # Fill template with random content
        format_dict = {}
        for key, values in self.content_pools.items():
            if f'{{{key[:-1]}}}' in template:  # Remove 's' for singular form
                format_dict[key[:-1]] = random.choice(values)
        
        # Handle numbered items (item1, item2, topic1, topic2, etc.)
        for i in range(1, 3):
            for base_key in ['item', 'topic', 'concept']:
                key = f'{base_key}{i}'
                if f'{{{key}}}' in template:
                    pool_key = f'{base_key}s'
                    if pool_key in self.content_pools:
                        format_dict[key] = random.choice(self.content_pools[pool_key])
        
        # Fill remaining placeholders with generic content
        remaining_keys = []
        import re
        for match in re.finditer(r'\{(\w+)\}', template):
            key = match.group(1)
            if key not in format_dict:
                remaining_keys.append(key)
        
        for key in remaining_keys:
            # Try to match with similar content
            if 'task' in key or 'activity' in key or 'procedure' in key:
                format_dict[key] = random.choice(self.content_pools['processes'])
            elif 'problem' in key or 'challenge' in key:
                format_dict[key] = f"optimizing {random.choice(self.content_pools['concepts'])} in {random.choice(self.content_pools['domains'])}"
            elif 'situation' in key or 'scenario' in key:
                format_dict[key] = random.choice(self.content_pools['scenarios'])
            elif 'setting' in key:
                format_dict[key] = f"a {random.choice(['modern', 'futuristic', 'innovative'])} {random.choice(self.content_pools['domains'])} environment"
            elif 'skill' in key:
                format_dict[key] = random.choice(self.content_pools['concepts'])
            else:
                format_dict[key] = random.choice(self.content_pools['topics'])
        
        try:
            return template.format(**format_dict)
        except KeyError:
            # Fallback if formatting fails
            return template.replace('{', '').replace('}', '')
    
    def generate_corpus_based_prompt(self) -> str:
        """Generate a prompt using the text corpus."""
        start_idx = random.randint(0, max(0, len(self.words) - 50))
        base_words = self.words[start_idx:start_idx + random.randint(20, 50)]
        
        prefixes = [
            "Analyze and expand on the following topic:",
            "Provide detailed insights about:",
            "Explain the significance of:",
            "Discuss the implications of:",
            "Elaborate on the concept of:",
            "Examine the following subject in depth:",
        ]
        
        prefix = random.choice(prefixes)
        content = " ".join(base_words)
        
        return f"{prefix} {content}. Include examples, current trends, and future implications."
    
    def generate_conversational_prompt(self) -> str:
        """Generate a conversational-style prompt."""
        starters = [
            "I'm curious about",
            "Can you help me understand",
            "I'd like to learn more about",
            "What are your thoughts on",
            "Could you explain",
            "I'm interested in exploring",
        ]
        
        topics = self.content_pools['topics'] + self.content_pools['concepts']
        topic = random.choice(topics)
        starter = random.choice(starters)
        
        follow_ups = [
            "Please provide a comprehensive overview.",
            "I'm particularly interested in practical applications.",
            "What are the current challenges and opportunities?",
            "How does this relate to recent developments?",
            "What should someone new to this field know?",
        ]
        
        follow_up = random.choice(follow_ups)
        
        return f"{starter} {topic}. {follow_up}"
    
    def generate_single_request(self, input_len: int, output_len: int) -> Tuple[str, int, int]:
        """Generate a single synthetic request."""
        # Choose generation method randomly
        methods = [
            self.generate_template_based_prompt,
            self.generate_corpus_based_prompt,
            self.generate_conversational_prompt,
        ]
        
        method = random.choice(methods)
        base_prompt = method()
        
        # Pad to desired length
        final_prompt = self.pad_to_length(base_prompt, input_len)
        
        return (final_prompt, input_len, output_len)
    
    def generate_requests(
        self,
        num_requests: int,
        input_len: int,
        output_len: int,
        variety: bool = True
    ) -> List[Tuple[str, int, int]]:
        """Generate multiple synthetic requests."""
        if not variety:
            # Simple fallback method
            requests = []
            for _ in range(num_requests):
                prompt = " ".join(["word"] * input_len)
                requests.append((prompt, input_len, output_len))
            return requests
        
        requests = []
        for _ in range(num_requests):
            request = self.generate_single_request(input_len, output_len)
            requests.append(request)
        
        return requests
    
    def save_to_file(
        self,
        requests: List[Tuple[str, int, int]],
        filename: str,
        format: str = 'json'
    ):
        """Save requests to file in specified format."""
        if format == 'json':
            data = []
            for i, (prompt, input_len, output_len) in enumerate(requests):
                data.append({
                    'id': i,
                    'prompt': prompt,
                    'input_length': input_len,
                    'output_length': output_len
                })
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == 'txt':
            with open(filename, 'w', encoding='utf-8') as f:
                for i, (prompt, input_len, output_len) in enumerate(requests):
                    f.write(f"=== Request {i+1} ===\n")
                    f.write(f"Input Length: {input_len} tokens\n")
                    f.write(f"Output Length: {output_len} tokens\n")
                    f.write(f"Prompt: {prompt}\n\n")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset for LLM inference testing')
    parser.add_argument('--num_requests', type=int, default=100, help='Number of requests to generate')
    parser.add_argument('--input_len', type=int, default=512, help='Input length in tokens')
    parser.add_argument('--output_len', type=int, default=256, help='Output length in tokens')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='Tokenizer to use (HuggingFace model name)')
    parser.add_argument('--output_file', type=str, default='synthetic_dataset.json', help='Output file path')
    parser.add_argument('--format', choices=['json', 'txt'], default='json', help='Output format')
    parser.add_argument('--simple', action='store_true', help='Use simple word repetition instead of realistic prompts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create generator
    generator = SyntheticDatasetGenerator(tokenizer)
    
    # Generate requests
    print(f"Generating {args.num_requests} requests with input_len={args.input_len}, output_len={args.output_len}")
    requests = generator.generate_requests(
        num_requests=args.num_requests,
        input_len=args.input_len,
        output_len=args.output_len,
        variety=not args.simple
    )
    
    # Save to file
    print(f"Saving to {args.output_file}")
    generator.save_to_file(requests, args.output_file, args.format)
    
    print(f"Successfully generated {len(requests)} synthetic requests")
    
    # Print sample
    if requests:
        print("\n=== Sample Request ===")
        sample_prompt, sample_input_len, sample_output_len = requests[0]
        print(f"Input Length: {sample_input_len} tokens")
        print(f"Output Length: {sample_output_len} tokens")
        print(f"Prompt preview: {sample_prompt[:200]}...")


if __name__ == "__main__":
    main()
