import gradio as gr
from segmentation import predict

THRESHOLD = 89.85

def run_prediction(contract, tenure, monthly):
    segment, action, reason = predict(
        contract,
        tenure,
        monthly,
        THRESHOLD
    )
    return segment, action, reason


with gr.Blocks() as demo:
    gr.Markdown("Customer Segmentation & Action System")

    with gr.Row():
        contract = gr.Dropdown(
            ["Month-to-month", "One-year", "Two-year"],
            label="Contract Type"
        )

    with gr.Row():
        tenure = gr.Slider(0, 72, label="Tenure (ay)")
        monthly = gr.Number(minimum=18.25,maximum=118.75,label="Monthly Charges")

    btn = gr.Button("Analiz Et")

    segment_out = gr.Textbox(label="Segment")
    action_out = gr.Textbox(label="Önerilen Aksiyon")
    reason_out = gr.Textbox(label="Açıklama")

    btn.click(
        fn=run_prediction,
        inputs=[contract, tenure, monthly],
        outputs=[segment_out, action_out, reason_out]
    )

demo.launch()