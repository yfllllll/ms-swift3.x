from typing import Type

import gradio as gr

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING, ModelType
from swift.ui.base import BaseUI


class Model(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'model_type': {
            'label': {
                'zh': '选择模型',
                'en': 'Select Model'
            },
            'info': {
                'zh': 'SWIFT已支持的模型名称',
                'en': 'Base model supported by SWIFT'
            }
        },
        'model_id_or_path': {
            'label': {
                'zh': '模型id或路径',
                'en': 'Model id or path'
            },
            'info': {
                'zh': '实际的模型id',
                'en': 'The actual model id or model path'
            }
        },
        'template_type': {
            'label': {
                'zh': '模型Prompt模板类型',
                'en': 'Prompt template type'
            },
            'info': {
                'zh': '选择匹配模型的Prompt模板',
                'en': 'Choose the template type of the model'
            }
        },
        'system': {
            'label': {
                'zh': 'system字段',
                'en': 'system'
            },
            'info': {
                'zh': '选择system字段的内容',
                'en': 'Choose the content of the system field'
            }
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            model_type = gr.Dropdown(
                elem_id='model_type',
                choices=ModelType.get_model_name_list(),
                scale=20)
            model_id_or_path = gr.Textbox(
                elem_id='model_id_or_path',
                lines=1,
                scale=20,
                interactive=False)
            template_type = gr.Dropdown(
                elem_id='template_type',
                choices=list(TEMPLATE_MAPPING.keys()) + ['AUTO'],
                scale=20)
        with gr.Row():
            system = gr.Textbox(elem_id='system', lines=1, scale=20)

        def update_input_model(choice):
            if choice is None:
                return None, None, None
            model_id_or_path = MODEL_MAPPING[choice]['model_id_or_path']
            default_system = getattr(
                TEMPLATE_MAPPING[MODEL_MAPPING[choice]['template']]
                ['template'], 'default_system', None)
            template = MODEL_MAPPING[choice]['template']
            return model_id_or_path, default_system, template

        model_type.change(
            update_input_model,
            inputs=[model_type],
            outputs=[model_id_or_path, system, template_type])
