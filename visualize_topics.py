#!/usr/bin/env python3
import re
from graphviz import Digraph
import os

class TopicNode:
    def __init__(self, id=None, name=None, parent=None):
        self.id = id
        self.name = name
        self.parent = parent
        self.children = []
        self.terms = []  # Store representative terms for the topic

def parse_topic_nodes(file_path):
    topics = {}
    current_vid = None
    current_path = None
    current_topic = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Parse VID
        if line.startswith('VID :'):
            current_vid = line.split(':')[1].strip()
            i += 1
            continue
            
        # Parse TARGET PATH
        if line.startswith('TARGET PATH :'):
            current_path = line.split(':')[1].strip()
            path_parts = [p.strip() for p in current_path.split('->')]
            
            # Create nodes for path if they don't exist
            parent = None
            for part in path_parts:
                if part == '*':
                    continue
                if part not in topics:
                    topics[part] = TopicNode(name=part, parent=parent)
                    if parent:
                        parent.children.append(topics[part])
                parent = topics[part]
            i += 1
            continue
            
        # Parse Topic sections
        if line.startswith('Topic'):
            match = re.match(r'Topic\s+(\d+)\s*:\s*(.+)', line)
            if match:
                topic_id, topic_name = match.groups()
                node_id = f"{current_vid}_{topic_id}"
                current_topic = TopicNode(id=node_id, name=topic_name, parent=parent)
                topics[node_id] = current_topic
                if parent:
                    parent.children.append(current_topic)
                
                # Collect terms
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('Topic') and not lines[i].strip().startswith('VID'):
                    term = lines[i].strip().split(']')[-1].strip()
                    if term:
                        current_topic.terms.append(term)
                    i += 1
                continue
        i += 1
    
    return topics

def create_visualization(topics, output_file):
    dot = Digraph(comment='Topic Hierarchy')
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Node styling
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    # Add nodes and edges
    added_nodes = set()
    
    def add_node_and_children(node):
        if node.name not in added_nodes:
            # Create label with topic name and top terms
            label = node.name
            if node.terms:
                top_terms = node.terms[:3]  # Show top 3 terms
                label += '\n' + '\n'.join(top_terms)
            
            # Add node
            node_id = node.id if node.id else node.name
            dot.node(str(node_id), label)
            added_nodes.add(node.name)
            
            # Add edge from parent
            if node.parent:
                parent_id = node.parent.id if node.parent.id else node.parent.name
                dot.edge(str(parent_id), str(node_id))
            
            # Recursively add children
            for child in node.children:
                add_node_and_children(child)
    
    # Start with root nodes (nodes without parents)
    root_nodes = [node for node in topics.values() if not node.parent]
    for root in root_nodes:
        add_node_and_children(root)
    
    # Save the visualization
    dot.render(output_file, view=True, cleanup=True)

def main():
    input_file = 'new_topic_nodes.txt'
    output_file = 'topic_hierarchy'
    
    # Parse the topic nodes
    topics = parse_topic_nodes(input_file)
    
    # Create and save visualization
    create_visualization(topics, output_file)
    print(f"Visualization has been saved to {output_file}.pdf")

if __name__ == '__main__':
    main()
