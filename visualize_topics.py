#!/usr/bin/env python3
import json
from graphviz import Digraph
import argparse
from pathlib import Path

class TopicNode:
    def __init__(self, name=None, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.phrases = []  # Store representative phrases
        self.size = 0

def parse_topic_nodes(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    topics = {}
    root = TopicNode(name="root")
    topics["root"] = root
    
    # First pass: create all nodes
    for path, discovered_topics in data.items():
        path_parts = [p.strip() for p in path.split('->')]
        
        # Create nodes for path if they don't exist
        parent = None
        for part in path_parts:
            if part not in topics:
                topics[part] = TopicNode(name=part, parent=parent)
                if parent:
                    parent.children.append(topics[part])
            parent = topics[part]
            
        # Add discovered topics as children
        for topic in discovered_topics:
            node = TopicNode(
                name=topic["name"], 
                parent=parent
            )
            node.phrases = [p["text"] for p in topic["phrases"]]
            node.size = topic["size"]
            topics[topic["name"]] = node
            parent.children.append(node)
    
    return topics

def create_visualization(topics, output_file):
    dot = Digraph(comment='Topic Hierarchy')
    dot.attr(rankdir='TB')  # Top to bottom layout
    
    # Node styling
    dot.attr('node', shape='box', style='rounded,filled')
    
    # Add nodes and edges
    added_nodes = set()
    
    def add_node_and_children(node):
        if node.name not in added_nodes:
            # Create label with topic name and top phrases
            label = f"{node.name}"
            if hasattr(node, 'size') and node.size > 0:
                label += f"\n(size: {node.size})"
            if node.phrases:
                top_phrases = node.phrases[:3]  # Show top 3 phrases
                label += '\n' + '\n'.join(f'• {p}' for p in top_phrases)
            
            # Color nodes based on type
            fillcolor = 'lightblue' if not node.phrases else '#90EE90'  # Light green for discovered topics
            
            # Add node
            dot.node(node.name, label, fillcolor=fillcolor)
            added_nodes.add(node.name)
            
            # Add edge from parent
            if node.parent:
                dot.edge(node.parent.name, node.name)
            
            # Process children
            for child in node.children:
                add_node_and_children(child)
    
    # Start from root
    root = topics["root"]
    add_node_and_children(root)
    
    # Save visualization
    output_path = Path(output_file)
    dot.render(output_path.stem, directory=output_path.parent, format='pdf', cleanup=True)

def main():
    parser = argparse.ArgumentParser(description='Visualize topic hierarchy')
    parser.add_argument('input_file', help='Path to discovered_topics.json')
    parser.add_argument('--output', default='topic_hierarchy.pdf', help='Output PDF file path')
    args = parser.parse_args()
    
    topics = parse_topic_nodes(args.input_file)
    create_visualization(topics, args.output)
    print(f"Visualization saved to {args.output}")

if __name__ == '__main__':
    main()
