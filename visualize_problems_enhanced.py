#!/usr/bin/env python3
"""
Enhanced visualization for mathematical problems using Gradio with proper math rendering
"""

import json
import gradio as gr
import re
from typing import List, Dict, Tuple

# HTML template for math rendering
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
            margin: 0;
            padding: 20px;
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .metadata {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metadata span {{
            margin-right: 20px;
            color: #666;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .math {{ 
            margin: 10px 0; 
        }}
        .katex-display {{
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div id="content">{content}</div>
    <script>
        renderMathInElement(document.getElementById('content'), {{
            delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '$', right: '$', display: false}},
                {{left: '\\\\(', right: '\\\\)', display: false}},
                {{left: '\\\\[', right: '\\\\]', display: true}}
            ],
            throwOnError: false
        }});
    </script>
</body>
</html>
"""

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

def format_problem_html(problem: Dict) -> str:
    """Format a problem as HTML with proper math support"""
    # Create header with metadata
    html = f"<h1>{problem.get('id', 'Unknown ID')}</h1>\n"
    
    html += '<div class="metadata">\n'
    html += f'<span><strong>Contest:</strong> {problem.get("contest", "Unknown")}</span>\n'
    html += f'<span><strong>Year:</strong> {problem.get("year", "Unknown")}</span>\n'
    html += f'<span><strong>Problem Number:</strong> {problem.get("problem_number", "Unknown")}</span>\n'
    if problem.get('source_pdf'):
        html += f'<span><strong>Source:</strong> {problem.get("source_pdf")}</span>\n'
    html += '</div>\n'
    
    # Add problem section
    html += '<div class="section">\n'
    html += '<h2>Problem</h2>\n'
    problem_text = problem.get('problem', 'No problem text available')
    # Keep the original LaTeX delimiters - KaTeX will handle them
    html += f'<div>{problem_text}</div>\n'
    html += '</div>\n'
    
    # Add solution section
    html += '<div class="section">\n'
    html += '<h2>Solution</h2>\n'
    solution_text = problem.get('solution', 'No solution available')
    # Convert newlines to <br> for better formatting
    solution_text = solution_text.replace('\n', '<br>\n')
    html += f'<div>{solution_text}</div>\n'
    html += '</div>\n'
    
    return HTML_TEMPLATE.format(content=html)

def create_interface():
    """Create enhanced Gradio interface"""
    # Load problems
    problems = load_problems('pairs.jsonl')
    
    if not problems:
        return gr.Interface(
            fn=lambda: "No problems found in pairs.jsonl",
            inputs=[],
            outputs="text",
            title="Problem Viewer - Error"
        )
    
    # State to track current problem index
    current_index = gr.State(value=0)
    
    def navigate(direction: str, index: int) -> Tuple[str, int, str]:
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
        
        html_content = format_problem_html(problems[new_index])
        status = f"Problem {new_index + 1} of {len(problems)}"
        
        return html_content, new_index, status
    
    def search_problem(search_term: str, current: int) -> Tuple[str, int, str]:
        """Search for a problem by ID or content"""
        search_lower = search_term.lower()
        
        for i, problem in enumerate(problems):
            if (search_lower in problem.get('id', '').lower() or
                search_lower in problem.get('problem', '').lower() or
                search_lower in problem.get('solution', '').lower()):
                html_content = format_problem_html(problem)
                return html_content, i, f"Found: Problem {i + 1} of {len(problems)}"
        
        return format_problem_html(problems[current]), current, "No match found"
    
    def go_to_problem(problem_num: int, current: int) -> Tuple[str, int, str]:
        """Go to a specific problem number"""
        if 1 <= problem_num <= len(problems):
            new_index = problem_num - 1
            html_content = format_problem_html(problems[new_index])
            return html_content, new_index, f"Problem {problem_num} of {len(problems)}"
        else:
            return format_problem_html(problems[current]), current, f"Please enter a number between 1 and {len(problems)}"
    
    # Create the interface
    with gr.Blocks(title="Mathematical Problems Viewer (Enhanced)", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Mathematical Problems Viewer - Enhanced Edition")
        gr.Markdown("Navigate through mathematical problems with proper LaTeX rendering using KaTeX")
        
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
        
        # Use HTML component for proper math rendering
        problem_display = gr.HTML(
            value=format_problem_html(problems[0]),
            label="Problem Display"
        )
        
        with gr.Row():
            first_btn = gr.Button("⏮️ First", variant="secondary")
            prev_btn = gr.Button("◀️ Previous", variant="secondary")
            next_btn = gr.Button("Next ▶️", variant="primary")
            last_btn = gr.Button("Last ⏭️", variant="secondary")
        
        # Set up event handlers
        first_btn.click(
            lambda i: navigate("first", i),
            inputs=[current_index],
            outputs=[problem_display, current_index, status_text]
        )
        
        prev_btn.click(
            lambda i: navigate("prev", i),
            inputs=[current_index],
            outputs=[problem_display, current_index, status_text]
        )
        
        next_btn.click(
            lambda i: navigate("next", i),
            inputs=[current_index],
            outputs=[problem_display, current_index, status_text]
        )
        
        last_btn.click(
            lambda i: navigate("last", i),
            inputs=[current_index],
            outputs=[problem_display, current_index, status_text]
        )
        
        search_btn.click(
            search_problem,
            inputs=[search_box, current_index],
            outputs=[problem_display, current_index, status_text]
        )
        
        search_box.submit(
            search_problem,
            inputs=[search_box, current_index],
            outputs=[problem_display, current_index, status_text]
        )
        
        go_btn.click(
            go_to_problem,
            inputs=[problem_number, current_index],
            outputs=[problem_display, current_index, status_text]
        )
        
        problem_number.submit(
            go_to_problem,
            inputs=[problem_number, current_index],
            outputs=[problem_display, current_index, status_text]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=8000,
        show_error=True
    ) 