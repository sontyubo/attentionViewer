import gradio as gr

if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Input Image")
                view_btn = gr.Button("View Attention")
            with gr.Column():
                attention_img = gr.Image(type="pil", label="Attention Image")

        view_btn.click(fn=lambda x: x, inputs=[input_img], outputs=[attention_img])

    demo.launch()
