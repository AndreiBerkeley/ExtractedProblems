#!/usr/bin/env python3
"""
Visualize mathematical problems from JSONL file using Gradio
"""

import json
import gradio as gr
import re
from typing import List, Dict, Tuple

def load_problems(file_path: str) -> List[Dict]:
    """Load problems from JSONL file"""
    problems = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
    except Exception as e:
        print(f"Error loading file: {e}")
    return problems

def convert_math_delimiters(text: str) -> str:
    """Convert LaTeX delimiters: \( \) → $...$ and \[ \] → $$...$$ for Markdown/MathJax"""
    if not text:
        return ""
    
    import re
    
    # Convert display math \\[ ... \\] to block $$ ... $$
    display_pattern = r'\\\[(.*?)\\\]'
    text = re.sub(
        display_pattern,
        lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n",
        text,
        flags=re.DOTALL,
    )

    # Convert inline math \\( ... \\) to inline $ ... $
    inline_pattern = r'\\\((.*?)\\\)'
    text = re.sub(
        inline_pattern,
        lambda m: f"${m.group(1).strip()}$",
        text,
    )

    # Clean up excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def escape_angle_brackets_in_math(text: str) -> str:
    """Replace < and > with \lt and \gt inside math delimiters only.
    Supports $$...$$, \[...\], $...$, and \(...\).
    """
    if not text:
        return ""
    import re

    def transform(content: str) -> str:
        # Use TeX relation symbols; add minimal spacing to avoid token merging
        return content.replace('<', r' \lt ').replace('>', r' \gt ')
    
    # print(text)

    # Process block math first to avoid inner overlaps
    patterns = [
        (re.compile(r'\$\$(.*?)\$\$', re.DOTALL), '$$', '$$'),
        (re.compile(r'\\\[(.*?)\\\]', re.DOTALL), r'\[', r'\]'),
        (re.compile(r'\$(.*?)\$', re.DOTALL), '$', '$'),
        (re.compile(r'\\\((.*?)\\\)', re.DOTALL), r'\(', r'\)'),
    ]

    for pattern, left, right in patterns:
        def repl(m):
            inner = m.group(1)
            return f"{left}{transform(inner)}{right}"
        text = pattern.sub(repl, text)

    # print(text)

    return text

def format_problem_display(problem: Dict) -> Tuple[str, str, str]:
    """Format a problem for display"""
    # Create header with metadata
    header = f"# {problem.get('id', 'Unknown ID')}\n\n"
    header += f"**Contest:** {problem.get('contest', 'Unknown')}\n"
    header += f"**Year:** {problem.get('year', 'Unknown')}\n"
    header += f"**Problem Number:** {problem.get('problem_number', 'Unknown')}\n"
    if problem.get('source_pdf'):
        header += f"**Source:** {problem.get('source_pdf')}\n"
    
    # Format problem statement
    problem_text = problem.get('problem', 'No problem text available')
    problem_text = escape_angle_brackets_in_math(problem_text)
    # problem_text = convert_math_delimiters(problem.get('problem', 'No problem text available'))
    problem_display = f"## Problem\n\n{problem_text}"
    
    # Format solution
    solution_text = problem.get('solution', 'No solution available')
    solution_text = escape_angle_brackets_in_math(solution_text)
    # solution_text = convert_math_delimiters(problem.get('solution', 'No solution available'))
    solution_display = f"## Solution\n\n{solution_text}"
    
    return header, problem_display, solution_display

def create_interface():
    """Create Gradio interface"""
    # Load problems
    problems = load_problems('pairs.jsonl')
    
    if not problems:
        return gr.Interface(
            fn=lambda: "No problems found in pairs.jsonl",
            inputs=[],
            outputs="text",
            title="Problem Viewer - Error"
        )
    
    # current_index component is created inside the Blocks UI
    
    def navigate(direction: str, index: int) -> Tuple[str, str, str, int, str]:
        """Navigate between problems"""
        if direction == "next":
            new_index = min(index + 1, len(problems) - 1)
        elif direction == "prev":
            new_index = max(index - 1, 0)
        elif direction == "first":
            new_index = 0
        elif direction == "last":
            new_index = len(problems) - 1
        else:
            new_index = index
        
        header, problem, solution = format_problem_display(problems[new_index])
        status = f"Problem {new_index + 1} of {len(problems)}"
        
        return header, problem, solution, new_index, status
    
    def search_problem(search_term: str, current: int) -> Tuple[str, str, str, int, str]:
        """Search for a problem by ID or content"""
        search_lower = search_term.lower()
        
        for i, problem in enumerate(problems):
            # Search in ID, problem text, and solution
            if (search_lower in problem.get('id', '').lower() or
                search_lower in problem.get('problem', '').lower() or
                search_lower in problem.get('solution', '').lower()):
                header, prob, sol = format_problem_display(problem)
                return header, prob, sol, i, f"Found: Problem {i + 1} of {len(problems)}"
        
        return "No match found", "", "", current, "No match found"
    
    def go_to_problem(problem_num: int, current: int) -> Tuple[str, str, str, int, str]:
        """Go to a specific problem number"""
        if 1 <= problem_num <= len(problems):
            new_index = problem_num - 1
            header, problem, solution = format_problem_display(problems[new_index])
            return header, problem, solution, new_index, f"Problem {problem_num} of {len(problems)}"
        else:
            return "Invalid problem number", "", "", current, f"Please enter a number between 1 and {len(problems)}"
    
    # Create the interface
    with gr.Blocks(title="Mathematical Problems Viewer", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Mathematical Problems Viewer")
        gr.Markdown("Navigate through mathematical problems with proper LaTeX rendering")
        
        with gr.Row():
            with gr.Column(scale=2):
                search_box = gr.Textbox(
                    label="Search",
                    placeholder="Search by ID or content...",
                    scale=2
                )
            with gr.Column(scale=1):
                search_btn = gr.Button("Search", variant="secondary")
            with gr.Column(scale=1):
                problem_number = gr.Number(
                    label="Go to Problem #",
                    minimum=1,
                    maximum=len(problems),
                    step=1,
                    value=1
                )
            with gr.Column(scale=1):
                go_btn = gr.Button("Go", variant="secondary")
        
        status_text = gr.Markdown(f"Problem 1 of {len(problems)}")
        
        # Hidden current index component
        current_index = gr.Number(value=0, visible=False)
        
        with gr.Row():
            header_display = gr.Markdown()
        
        with gr.Row():
            problem_display = gr.Markdown(
                label="Problem",
                elem_id="problem-display",
                latex_delimiters=[ 
                    {"left": "$", "right": "$", "display": False },
                    {"left": "$$", "right": "$$", "display": True },
                    {"left": "\\(", "right": "\\)", "display": False },
                    {"left": "\\[", "right": "\\]", "display": True }
                ]
            )
        
        with gr.Row():
            solution_display = gr.Markdown(
                label="Solution",
                elem_id="solution-display",
                latex_delimiters=[ 
                    {"left": "$", "right": "$", "display": False },
                    {"left": "$$", "right": "$$", "display": True },
                    {"left": "\\(", "right": "\\)", "display": False },
                    {"left": "\\[", "right": "\\]", "display": True }
                ]
            )
        
        with gr.Row():
            first_btn = gr.Button("⏮️ First", variant="secondary")
            prev_btn = gr.Button("◀️ Previous", variant="secondary")
            next_btn = gr.Button("Next ▶️", variant="primary")
            last_btn = gr.Button("Last ⏭️", variant="secondary")
        
        # Initialize with first problem
        initial_header, initial_problem, initial_solution = format_problem_display(problems[0])
        header_display.value = initial_header
        problem_display.value = initial_problem
        solution_display.value = initial_solution
        
        # Set up event handlers
        first_btn.click(
            lambda i: navigate("first", i),
            inputs=[current_index],
            outputs=[header_display, problem_display, solution_display, current_index, status_text]
        )
        
        prev_btn.click(
            lambda i: navigate("prev", i),
            inputs=[current_index],
            outputs=[header_display, problem_display, solution_display, current_index, status_text]
        )
        
        next_btn.click(
            lambda i: navigate("next", i),
            inputs=[current_index],
            outputs=[header_display, problem_display, solution_display, current_index, status_text]
        )
        
        last_btn.click(
            lambda i: navigate("last", i),
            inputs=[current_index],
            outputs=[header_display, problem_display, solution_display, current_index, status_text]
        )
        
        search_btn.click(
            search_problem,
            inputs=[search_box, current_index],
            outputs=[header_display, problem_display, solution_display, current_index, status_text]
        )
        
        search_box.submit(
            search_problem,
            inputs=[search_box, current_index],
            outputs=[header_display, problem_display, solution_display, current_index, status_text]
        )
        
        go_btn.click(
            go_to_problem,
            inputs=[problem_number, current_index],
            outputs=[header_display, problem_display, solution_display, current_index, status_text]
        )
        
        problem_number.submit(
            go_to_problem,
            inputs=[problem_number, current_index],
            outputs=[header_display, problem_display, solution_display, current_index, status_text]
        )
        
        # Add custom CSS for better math rendering
        interface.css = """
        #problem-display, #solution-display {
            font-size: 16px;
            line-height: 1.8;
        }
        .math {
            margin: 10px 0;
        }
        """
        
        # Load initial problem
        interface.load(
            lambda: (initial_header, initial_problem, initial_solution, 0, f"Problem 1 of {len(problems)}"),
            outputs=[header_display, problem_display, solution_display, current_index, status_text]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        show_error=True
    ) 