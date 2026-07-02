import gradio as gr
from app import build_interface

demo = build_interface()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_port=7867)
