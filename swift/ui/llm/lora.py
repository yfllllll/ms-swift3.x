import gradio as gr

from swift.llm import MODEL_MAPPING
from swift.ui.element import elements
from swift.ui.i18n import get_i18n_labels


def lora():
    get_i18n_labels(i18n)
    with gr.Accordion(elem_id='lora_tab', open=True):
        with gr.Blocks():
            with gr.Row():
                lora_target_modules = gr.Textbox(
                    elem_id='lora_target_modules',
                    lines=1,
                    scale=20,
                    is_list=True)
            with gr.Row():
                gr.Slider(
                    elem_id='lora_rank',
                    value=32,
                    minimum=1,
                    maximum=512,
                    step=8)
                gr.Slider(
                    elem_id='lora_alpha',
                    value=8,
                    minimum=1,
                    maximum=512,
                    step=8)
                gr.Textbox(elem_id='lora_dropout_p')

        def update_lora(choice):
            return ' '.join(MODEL_MAPPING[choice]['lora_target_modules'])

        elements['model_type'].change(
            update_lora,
            inputs=[elements['model_type']],
            outputs=[lora_target_modules])


i18n = {
    'lora_tab': {
        'label': {
            'zh': 'LoRA参数设置',
            'en': 'LoRA settings'
        },
    },
    'lora_target_modules': {
        'label': {
            'zh': 'LoRA目标模块',
            'en': 'LoRA target modules'
        },
        'info': {
            'zh': '设置LoRA目标模块，如训练所有Linear请改为ALL',
            'en':
            'Set the LoRA target modules, fill in ALL if train all Linears'
        }
    },
    'lora_rank': {
        'label': {
            'zh': 'LoRA的秩',
            'en': 'The LoRA rank'
        }
    },
    'lora_alpha': {
        'label': {
            'zh': 'LoRA的alpha',
            'en': 'The LoRA alpha'
        }
    },
    'lora_dropout_p': {
        'label': {
            'zh': 'LoRA的dropout',
            'en': 'The LoRA dropout'
        }
    },
}
