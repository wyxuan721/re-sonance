import os
import time
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    pipeline
from zhconv import convert
from openai import OpenAI
import gradio as gr
import dashscope
from dashscope.audio.tts_v2 import *

# 配置API Key
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", "******"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 设置DashScope API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "******")

# Whisper模型配置
pretrained_model_id = "openai/whisper-large-v3-turbo"
model_id = "openai/whisper-large-v3-turbo"
processor_cache_dir = "./processor_cache"
model_cache_dir = "./model_cache"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 初始化Whisper模型和处理器
feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_id)
tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_id, language="Chinese", task="transcribe",
                                             cache_dir=processor_cache_dir)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
model = WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir=model_cache_dir)

pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
)


def transcribe_speech(filepath):
    output = pipe(
        filepath,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe",
            "language": "chinese",
        },
        chunk_length_s=30,
        batch_size=1,
    )
    return convert(output["text"], 'zh-cn')


def optimize_with_llm(text):
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system",
                 "content": "你是一个你是一名语言学专家，负责优化由构音障碍语音生成的初步文本，使其更接近真实语义。你将收到一段文本，这段文本是由构音障碍患者发音转录的。请根据语法和上下文，纠正可能的错误，使其更加流畅和自然。"},
                {"role": "user", "content": f"""
                要求：
                1.除了可以删除多个连续的字外，不要增删字的个数，以免扩大错误率；
                2.只需要输出调整后的句子或词，不需要输出"调整后的文本"、"这句话不太通顺"此类提示。
                3.发现逻辑不通的词时，重点根据近音词联想，不要随便调换词语的位置等。
                4.有的句子本身就是通顺、符合逻辑的，那么则不需要修改，直接输出即可。
                例子1：
                    原文：可能易挖房价泡沫奉献,在积极深入放缓间段入货币侦测工具时,基本住房需求得到满足时后,对绿色高教依据的高频率住房需求快速上升
                    优化后的文本：可能引发房价泡沫风险 ，在经济增速放缓阶段运用货币政策工具时 ，基本住房需求得到满足后 ，对绿色高效宜居的高品质住房需求快速上升
                在这个例子中，易挖和引发声音接近，你应该想到引发更贴合语境；绿色高效宜居比绿色高教依据更贴合语境，因此应该加以优化。
                例子2：
                    原文：联邦学习效果非常好
                    优化后的文本：联邦学习效果非常好
                在这个例子中，原文已经比较通顺，词语已经比较符合逻辑，则不需要修改，直接输出即可。
                现在我将给你原文，请你给出优化后的文本，严禁给出任何分析过程，不需要给我任何提示！只需要给我优化后的文本！不要给出任何表达建议！
                原文：{text}
                直接给出优化后的文本，不必给出任何分析过程："""}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        return text


def generate_speech(text, voice="longxiaochun"):
    try:
        output_dir = "generated_audio"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"tts_output_{timestamp}.wav")

        synthesizer = SpeechSynthesizer(
            model="cosyvoice-v1",
            voice=voice,
            format=AudioFormat.WAV_24000HZ_MONO_16BIT
        )

        audio = synthesizer.call(text)
        with open(output_file, 'wb') as f:
            f.write(audio)

        return output_file if os.path.getsize(output_file) > 44 else None

    except Exception as e:
        print(f"Error during speech generation: {str(e)}")
        return None


def create_streaming_interface():
    voice_choices = ["longxiaochun", "longwan", "longxiaocheng", "longshu"]
    audio_entries = []

    class AudioEntry:
        def __init__(self, original_audio, transcription, optimized_text, synthesized_audio):
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.original_audio = original_audio
            self.transcription = transcription
            self.optimized_text = optimized_text
            self.synthesized_audio = synthesized_audio

    def process_audio(audio_path, selected_voice, auto_play):
        if audio_path is None:
            return "Please record audio first", [], None, ""

        try:
            transcription = transcribe_speech(audio_path)
            optimized = optimize_with_llm(transcription)
            synthesized_path = generate_speech(optimized, voice=selected_voice)

            entry = AudioEntry(audio_path, transcription, optimized, synthesized_path)
            audio_entries.append(entry)

            entries_display = [[
                idx,
                entry.timestamp,
                entry.transcription,
                entry.optimized_text
            ] for idx, entry in enumerate(audio_entries)]

            if auto_play:
                return "Processing completed", entries_display, synthesized_path, f"Now playing: {optimized}"
            else:
                return "Processing completed", entries_display, None, ""

        except Exception as e:
            return f"Processing error: {str(e)}", [], None, ""

    def play_selected_entry(evt: gr.SelectData):
        if evt.index[0] < len(audio_entries):
            selected_entry = audio_entries[evt.index[0]]
            return {
                output_audio: selected_entry.synthesized_audio,
                play_status: f"Now playing: {selected_entry.optimized_text}"
            }
        return {
            output_audio: None,
            play_status: "Invalid selection"
        }

    def auto_refresh():
        entries_display = [[
            idx,
            entry.timestamp,
            entry.transcription,
            entry.optimized_text
        ] for idx, entry in enumerate(audio_entries)]
        return entries_display

    def clear_audio():
        return None

    def clear_history(clear_type, num_entries):
        if clear_type == "Clear All":
            audio_entries.clear()
        elif clear_type == "Clear Recent N" and num_entries > 0:
            del audio_entries[-min(num_entries, len(audio_entries)):]

        entries_display = [[
            idx,
            entry.timestamp,
            entry.transcription,
            entry.optimized_text
        ] for idx, entry in enumerate(audio_entries)]

        return entries_display, f"{clear_type} completed, {len(audio_entries)} entries remaining"

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Input Area")
                audio_input = gr.Audio(
                    sources=None,
                    type="filepath",
                    label="Click to Record",
                    format="wav",
                    streaming=False,
                    interactive=True,
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear Recording")
                    process_btn = gr.Button("Process")
                voice_selector = gr.Dropdown(
                    choices=voice_choices,
                    value="longxiaochun",
                    label="Select Voice"
                )
                auto_play_switch = gr.Checkbox(
                    label="Auto-play Results",
                    value=True,
                    info="Automatically play processed audio when ready"
                )
                status_output = gr.Textbox(label="Status")

            with gr.Column(scale=2):
                gr.Markdown("## Conversation History")
                with gr.Row():
                    clear_type = gr.Radio(
                        choices=["Clear All", "Clear Recent N"],
                        label="Clear Method",
                        value="Clear All"
                    )
                    clear_num = gr.Number(
                        value=1,
                        label="Number of Entries",
                        minimum=1,
                        step=1,
                        interactive=True
                    )
                    clear_history_btn = gr.Button("Clear History")

                conversation_history = gr.Dataframe(
                    headers=["No.", "Time", "Original Transcription", "Optimized Text"],
                    row_count=10,
                    col_count=(4, "fixed"),
                    interactive=False,
                    every=1
                )
                play_status = gr.Textbox(label="Now Playing", interactive=False)
                output_audio = gr.Audio(label="Audio Player", autoplay=True)

        process_btn.click(
            fn=process_audio,
            inputs=[audio_input, voice_selector, auto_play_switch],
            outputs=[status_output, conversation_history, output_audio, play_status]
        )

        clear_btn.click(
            fn=clear_audio,
            outputs=[audio_input]
        )

        clear_history_btn.click(
            fn=clear_history,
            inputs=[clear_type, clear_num],
            outputs=[conversation_history, status_output]
        )

        conversation_history.select(
            fn=play_selected_entry,
            outputs=[output_audio, play_status],
        )

        demo.load(auto_refresh, None, conversation_history, every=1)

    return demo

if __name__ == "__main__":
    if not os.path.exists("generated_audio"):
        os.makedirs("generated_audio")

    demo = create_streaming_interface()
    demo.queue().launch(
        share=True,
        show_error=True,
        server_name="localhost",
        server_port=7860,
        ssl_verify=False
    )