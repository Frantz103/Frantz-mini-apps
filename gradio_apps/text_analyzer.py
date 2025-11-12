"""
Text Analyzer Demo
A lightweight text processing interface using Gradio.
"""

import gradio as gr
import re
from collections import Counter


def analyze_text(text):
    """
    Analyze text and return various statistics.
    """
    if not text or text.strip() == "":
        return {
            "summary": "No text provided",
            "stats": "",
            "word_freq": "",
        }

    # Basic statistics
    char_count = len(text)
    char_count_no_spaces = len(text.replace(" ", ""))
    word_count = len(text.split())
    sentence_count = len(re.split(r'[.!?]+', text)) - 1
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])

    # Word frequency
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    top_words = word_freq.most_common(10)

    # Build summary
    summary = f"""
## Text Analysis Summary

**Character Count:** {char_count}
**Character Count (no spaces):** {char_count_no_spaces}
**Word Count:** {word_count}
**Sentence Count:** {sentence_count}
**Paragraph Count:** {paragraph_count}
**Average Word Length:** {char_count_no_spaces / word_count if word_count > 0 else 0:.2f}
    """.strip()

    # Statistics table
    stats = f"""
| Metric | Value |
|--------|-------|
| Characters | {char_count} |
| Words | {word_count} |
| Sentences | {sentence_count} |
| Paragraphs | {paragraph_count} |
    """.strip()

    # Word frequency table
    word_freq_table = "| Word | Frequency |\n|------|----------|\n"
    for word, freq in top_words:
        word_freq_table += f"| {word} | {freq} |\n"

    return summary, stats, word_freq_table


# Create Gradio interface
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 📝 Text Analyzer")
    gr.Markdown("Enter text below to analyze its statistics and word frequency.")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter your text here...",
                lines=10,
            )
            analyze_btn = gr.Button("Analyze Text", variant="primary")

        with gr.Column():
            summary_output = gr.Markdown(label="Summary")

    with gr.Row():
        stats_output = gr.Markdown(label="Statistics")
        word_freq_output = gr.Markdown(label="Top 10 Words")

    analyze_btn.click(
        fn=analyze_text,
        inputs=text_input,
        outputs=[summary_output, stats_output, word_freq_output],
    )

    # Examples
    gr.Examples(
        examples=[
            ["The quick brown fox jumps over the lazy dog. The dog was really lazy."],
            ["Machine learning is a subset of artificial intelligence. It enables computers to learn from data."],
        ],
        inputs=text_input,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
    )
