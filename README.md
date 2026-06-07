# Vid2Blog (Formerly Youtube-video-to-blog-page-creator)

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![OpenAI](https://img.shields.io/badge/AI-OpenAI%20%7C%20Whisper-orange)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## About the Project

**Vid2Blog** is an automated AI pipeline that transforms any public YouTube video into a well-structured, SEO-friendly blog post. By extracting the video transcript and leveraging Large Language Models (LLMs), this tool synthesizes spoken content into highly readable articles, complete with headings, bullet points, and summaries.

Whether you are a content creator looking to repurpose your videos or a developer exploring audio-to-text pipelines, Vid2Blog provides a seamless, modular architecture for rapid content generation.

### Core Features
* **Automated Extraction:** Seamlessly downloads YouTube audio/transcripts.
* **Intelligent Transcription:** Utilizes advanced speech-to-text models for high accuracy.
* **Contextual Formatting:** Prompts an LLM to rewrite transcripts into engaging blog formats, removing filler words and structuring the narrative.
* **Markdown Export:** Outputs clean `.md` files ready to be pasted into WordPress, Ghost, or static site generators.

## Prerequisites

* Python 3.9+
* FFmpeg (required for local audio extraction)
* API Keys (e.g., OpenAI or Groq) if using cloud-based generation

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Harsh-7612/Vid2Blog.git](https://github.com/Harsh-7612/Vid2Blog.git)
   cd Vid2Blog
