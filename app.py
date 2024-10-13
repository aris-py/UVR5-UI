import os
import re
import uuid
import subprocess
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import gradio as gr
import yt_dlp
from scipy.io.wavfile import write, read

from tabs.settings import select_themes_tab
from assets.i18n.i18n import I18nAuto
import assets.themes.loadThemes as loadThemes

i18n = I18nAuto()

# Model Definitions
ROFORMER_MODELS = {
    'BS-Roformer-Viperx-1297.ckpt': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
    'BS-Roformer-Viperx-1296.ckpt': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
    'BS-Roformer-Viperx-1053.ckpt': 'model_bs_roformer_ep_937_sdr_10.5309.ckpt',
    'Mel-Roformer-Viperx-1143.ckpt': 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt',
    'BS-Roformer-De-Reverb-Anvuew': 'deverb_bs_roformer_8_384dim_10depth.ckpt',
    'Mel-Roformer-Crowd-Aufr33-Viperx': 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
    'Mel-Roformer-Denoise-Aufr33': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
    'Mel-Roformer-Denoise-Aufr33-Aggr': 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
    'Mel-Roformer-Karaoke-Aufr33-Viperx': 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'
}

MDX23C_MODELS = [
    'MDX23C_D1581.ckpt',
    'MDX23C-8KFFT-InstVoc_HQ.ckpt',
    'MDX23C-8KFFT-InstVoc_HQ_2.ckpt',
]

MDXNET_MODELS = [
    'UVR-MDX-NET-Inst_full_292.onnx',
    'UVR-MDX-NET_Inst_187_beta.onnx',
    'UVR-MDX-NET_Inst_82_beta.onnx',
    'UVR-MDX-NET_Inst_90_beta.onnx',
    'UVR-MDX-NET_Main_340.onnx',
    'UVR-MDX-NET_Main_390.onnx',
    'UVR-MDX-NET_Main_406.onnx',
    'UVR-MDX-NET_Main_427.onnx',
    'UVR-MDX-NET_Main_438.onnx',
    'UVR-MDX-NET-Inst_HQ_1.onnx',
    'UVR-MDX-NET-Inst_HQ_2.onnx',
    'UVR-MDX-NET-Inst_HQ_3.onnx',
    'UVR-MDX-NET-Inst_HQ_4.onnx',
    'UVR_MDXNET_Main.onnx',
    'UVR-MDX-NET-Inst_Main.onnx',
    'UVR_MDXNET_1_9703.onnx',
    'UVR_MDXNET_2_9682.onnx',
    'UVR_MDXNET_3_9662.onnx',
    'UVR-MDX-NET-Inst_1.onnx',
    'UVR-MDX-NET-Inst_2.onnx',
    'UVR-MDX-NET-Inst_3.onnx',
    'UVR_MDXNET_KARA.onnx',
    'UVR_MDXNET_KARA_2.onnx',
    'UVR_MDXNET_9482.onnx',
    'UVR-MDX-NET-Voc_FT.onnx',
    'Kim_Vocal_1.onnx',
    'Kim_Vocal_2.onnx',
    'Kim_Inst.onnx',
    'Reverb_HQ_By_FoxJoy.onnx',
    'UVR-MDX-NET_Crowd_HQ_1.onnx',
    'kuielab_a_vocals.onnx',
    'kuielab_a_other.onnx',
    'kuielab_a_bass.onnx',
    'kuielab_a_drums.onnx',
    'kuielab_b_vocals.onnx',
    'kuielab_b_other.onnx',
    'kuielab_b_bass.onnx',
    'kuielab_b_drums.onnx',
]

VRARCH_MODELS = [
    '1_HP-UVR.pth',
    '2_HP-UVR.pth',
    '3_HP-Vocal-UVR.pth',
    '4_HP-Vocal-UVR.pth',
    '5_HP-Karaoke-UVR.pth',
    '6_HP-Karaoke-UVR.pth',
    '7_HP2-UVR.pth',
    '8_HP2-UVR.pth',
    '9_HP2-UVR.pth',
    '10_SP-UVR-2B-32000-1.pth',
    '11_SP-UVR-2B-32000-2.pth',
    '12_SP-UVR-3B-44100.pth',
    '13_SP-UVR-4B-44100-1.pth',
    '14_SP-UVR-4B-44100-2.pth',
    '15_SP-UVR-MID-44100-1.pth',
    '16_SP-UVR-MID-44100-2.pth',
    '17_HP-Wind_Inst-UVR.pth',
    'UVR-De-Echo-Aggressive.pth',
    'UVR-De-Echo-Normal.pth',
    'UVR-DeEcho-DeReverb.pth',
    'UVR-DeNoise-Lite.pth',
    'UVR-DeNoise.pth',
    'UVR-BVE-4B_SN-44100-1.pth',
    'MGM_HIGHEND_v4.pth',
    'MGM_LOWEND_A_v4.pth',
    'MGM_LOWEND_B_v4.pth',
    'MGM_MAIN_v4.pth',
]

DEMUCUS_MODELS = [
    'htdemucs_ft.yaml',
    'htdemucs.yaml',
    'hdemucs_mmi.yaml',
]

OUTPUT_FORMATS = ['wav', 'flac', 'mp3']
MDXNET_OVERLAPS = ['0.25', '0.5', '0.75', '0.99']
VRARCH_WINDOW_SIZES = ['320', '512', '1024']
DEMUCUS_OVERLAPS = ['0.25', '0.50', '0.75', '0.99']

# Common Constants
OUTPUT_DIR = "./outputs"
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac")
NORMALIZATION = "0.9"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thread pool for batch processing
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

def generate_unique_id():
    return uuid.uuid4().hex[:5]

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join("ytdl", '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info_dict).rsplit('.', 1)[0] + '.wav'
            sample_rate, audio_data = read(file_path)
            return sample_rate, np.asarray(audio_data, dtype=np.int16)
        except Exception as e:
            print(f"Error downloading audio: {e}")
            return None, None

def run_separator(audio, model, output_format, additional_params):
    unique_id = generate_unique_id()
    input_path = os.path.join(OUTPUT_DIR, f'{unique_id}.wav')
    write(input_path, audio[0], audio[1])

    cmd = [
        "audio-separator",
        input_path,
        "-m", model,
        f"--output_dir={OUTPUT_DIR}",
        f"--output_format={output_format}",
        f"--normalization={NORMALIZATION}"
    ] + additional_params

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_path}: {e}")
        return [f"Error processing {input_path}: {str(e)}"]

    pattern = re.compile(unique_id)
    stem_files = sorted([
        os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if pattern.search(f)
    ])
    return stem_files

def roformer_separator(audio, model_key, output_format, overlap, segment_size):
    model = ROFORMER_MODELS.get(model_key)
    if not model:
        return ["Invalid Roformer model selected."]
    additional_params = [
        f"--mdxc_overlap={overlap}",
        f"--mdxc_segment_size={segment_size}"
    ]
    stems = run_separator(audio, model, output_format, additional_params)
    return stems[:2] if len(stems) >=2 else stems

def mdxc_separator(audio, model, output_format, segment_size, overlap, denoise):
    additional_params = [
        f"--mdxc_segment_size={segment_size}",
        f"--mdxc_overlap={overlap}"
    ]
    if denoise:
        additional_params.append("--mdx_enable_denoise")
    stems = run_separator(audio, model, output_format, additional_params)
    return stems[:2] if len(stems) >=2 else stems

def mdxnet_separator(audio, model, output_format, segment_size, overlap, denoise):
    additional_params = [
        f"--mdx_segment_size={segment_size}",
        f"--mdx_overlap={overlap}"
    ]
    if denoise:
        additional_params.append("--mdx_enable_denoise")
    stems = run_separator(audio, model, output_format, additional_params)
    return stems[:2] if len(stems) >=2 else stems

def vrarch_separator(audio, model, output_format, window_size, aggression, tta, high_end_process):
    additional_params = [
        f"--vr_window_size={window_size}",
        f"--vr_aggression={aggression}"
    ]
    if tta:
        additional_params.append("--vr_enable_tta")
    if high_end_process:
        additional_params.append("--vr_high_end_process")
    stems = run_separator(audio, model, output_format, additional_params)
    return stems[:2] if len(stems) >=2 else stems

def demucs_separator(audio, model, output_format, shifts, overlap):
    additional_params = [
        f"--demucs_shifts={shifts}",
        f"--demucs_overlap={overlap}"
    ]
    stems = run_separator(audio, model, output_format, additional_params)
    return tuple(stems[:4]) if len(stems) >=4 else tuple(stems)

def batch_separator(input_path, output_path, model, output_format, overlap, segment_size, denoise=None, window_size=None, aggression=None, tta=None, high_end_process=None):
    found_files = sorted([
        f for f in os.listdir(input_path) if f.endswith(AUDIO_EXTENSIONS)
    ])
    total_files = len(found_files)
    logs = []

    if total_files == 0:
        logs.append("No valid audio files.")
        return "\n".join(logs)
    
    logs.append(f"{total_files} audio files found.")
    for audio_file in found_files:
        file_path = os.path.join(input_path, audio_file)
        cmd = [
            "audio-separator",
            file_path,
            "-m", model,
            f"--output_dir={output_path}",
            f"--output_format={output_format}",
            f"--normalization={NORMALIZATION}"
        ]

        # Add additional parameters based on model type
        if "--mdxc_overlap" in model:
            cmd.append(f"--mdxc_overlap={overlap}")
            cmd.append(f"--mdxc_segment_size={segment_size}")
            if denoise:
                cmd.append("--mdx_enable_denoise")
        elif "--vr_window_size" in model:
            cmd.append(f"--vr_window_size={window_size}")
            cmd.append(f"--vr_aggression={aggression}")
            if tta:
                cmd.append("--vr_enable_tta")
            if high_end_process:
                cmd.append("--vr_high_end_process")
        # Add other model-specific parameters as needed

        logs.append(f"Processing file: {audio_file}")
        try:
            subprocess.run(cmd, check=True)
            logs.append(f"File: {audio_file} processed.")
        except subprocess.CalledProcessError as e:
            logs.append(f"Error processing {audio_file}: {str(e)}")
    return "\n".join(logs)

def select_themes():
    themes_select = gr.Dropdown(
        choices=loadThemes.get_list(),
        value=loadThemes.read_json(),
        label=i18n("Theme"),
        info=i18n("Select the theme you want to use. (Requires restarting the App)"),
        visible=True,
    )
    themes_select.change(
        fn=loadThemes.select_theme,
        inputs=themes_select,
        outputs=[],
    )

with gr.Blocks(theme=loadThemes.load_json() or "NoCrypt/miku", title="🎵 UVR5 UI 🎵") as app:
    gr.Markdown("<h1> 🎵 UVR5 UI 🎵 </h1>")
    gr.Markdown("If you like UVR5 UI, you can star my repo on [GitHub](https://github.com/Eddycrack864/UVR5-UI)")
    gr.Markdown("Try UVR5 UI on Hugging Face with A100 [here](https://huggingface.co/spaces/TheStinger/UVR5_UI)")
    
    with gr.Tabs():
        # BS/Mel Roformer Tab
        with gr.TabItem("BS/Mel Roformer"):
            with gr.Row():
                roformer_model = gr.Dropdown(
                    label="Select the Model",
                    choices=list(ROFORMER_MODELS.keys()),
                    interactive=True
                )
                roformer_output_format = gr.Dropdown(
                    label="Select the Output Format",
                    choices=OUTPUT_FORMATS,
                    interactive=True
                )
            with gr.Row():
                roformer_overlap = gr.Slider(
                    minimum=2,
                    maximum=4,
                    step=1,
                    label="Overlap",
                    info="Amount of overlap between prediction windows.",
                    value=4,
                    interactive=True
                )
                roformer_segment_size = gr.Slider(
                    minimum=32,
                    maximum=4000,
                    step=32,
                    label="Segment Size",
                    info="Larger consumes more resources, but may give better results.",
                    value=256,
                    interactive=True
                )
            with gr.Row():
                roformer_audio = gr.Audio(
                    label="Input Audio",
                    type="numpy",
                    interactive=True
                )
            with gr.Accordion("Separation by Link", open=False):
                with gr.Row():
                    roformer_link = gr.Textbox(
                        label="Link",
                        placeholder="Paste the link here",
                        interactive=True
                    )
                with gr.Row():
                    gr.Markdown("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)")
                with gr.Row():
                    roformer_download_button = gr.Button(
                        "Download!",
                        variant="primary"
                    )
            roformer_download_button.click(
                fn=download_audio,
                inputs=[roformer_link],
                outputs=[roformer_audio]
            )
            
            with gr.Accordion("Batch Separation", open=False):
                with gr.Row():
                    roformer_input_path = gr.Textbox(
                        label="Input Path",
                        placeholder="Place the input path here",
                        interactive=True
                    )
                    roformer_output_path = gr.Textbox(
                        label="Output Path",
                        placeholder="Place the output path here",
                        interactive=True
                    )
                with gr.Row():
                    roformer_batch_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    roformer_info = gr.Textbox(
                        label="Output Information",
                        interactive=False
                    )
            roformer_batch_button.click(
                fn=batch_separator, 
                inputs=[
                    roformer_input_path, 
                    roformer_output_path, 
                    roformer_model, 
                    roformer_output_format, 
                    roformer_overlap, 
                    roformer_segment_size
                ],
                outputs=[roformer_info]
            )
            
            with gr.Row():
                roformer_button = gr.Button("Separate!", variant="primary")
            with gr.Row():
                roformer_stem1 = gr.Audio(
                    show_download_button=True,
                    interactive=False,
                    label="Stem 1",
                    type="filepath"
                )
                roformer_stem2 = gr.Audio(
                    show_download_button=True,
                    interactive=False,
                    label="Stem 2",
                    type="filepath"
                )
            roformer_button.click(
                fn=roformer_separator, 
                inputs=[
                    roformer_audio, 
                    roformer_model, 
                    roformer_output_format, 
                    roformer_overlap, 
                    roformer_segment_size
                ],
                outputs=[roformer_stem1, roformer_stem2]
            )
        
        # MDX-NET Tab
        with gr.TabItem("MDX-NET"):
            with gr.Row():
                mdxnet_model = gr.Dropdown(
                    label="Select the Model",
                    choices=MDXNET_MODELS,
                    interactive=True
                )
                mdxnet_output_format = gr.Dropdown(
                    label="Select the Output Format",
                    choices=OUTPUT_FORMATS,
                    interactive=True
                )
            with gr.Row():
                mdxnet_segment_size = gr.Slider(
                    minimum=32,
                    maximum=4000,
                    step=32,
                    label="Segment Size",
                    info="Larger consumes more resources, but may give better results.",
                    value=256,
                    interactive=True
                )
                mdxnet_overlap = gr.Dropdown(
                    label="Overlap",
                    choices=MDXNET_OVERLAPS,
                    value=MDXNET_OVERLAPS[0],
                    interactive=True
                )
                mdxnet_denoise = gr.Checkbox(
                    label="Denoise",
                    info="Enable denoising during separation.",
                    value=True,
                    interactive=True
                )
            with gr.Row():
                mdxnet_audio = gr.Audio(
                    label="Input Audio",
                    type="numpy",
                    interactive=True
                )
            with gr.Accordion("Separation by Link", open=False):
                with gr.Row():
                    mdxnet_link = gr.Textbox(
                        label="Link",
                        placeholder="Paste the link here",
                        interactive=True
                    )
                with gr.Row():
                    gr.Markdown("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)")
                with gr.Row():
                    mdxnet_download_button = gr.Button(
                        "Download!",
                        variant="primary"
                    )
            mdxnet_download_button.click(
                fn=download_audio,
                inputs=[mdxnet_link],
                outputs=[mdxnet_audio]
            )
            
            with gr.Accordion("Batch Separation", open=False):
                with gr.Row():
                    mdxnet_input_path = gr.Textbox(
                        label="Input Path",
                        placeholder="Place the input path here",
                        interactive=True
                    )
                    mdxnet_output_path = gr.Textbox(
                        label="Output Path",
                        placeholder="Place the output path here",
                        interactive=True
                    )
                with gr.Row():
                    mdxnet_batch_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    mdxnet_info = gr.Textbox(
                        label="Output Information",
                        interactive=False
                    )
            mdxnet_batch_button.click(
                fn=batch_separator, 
                inputs=[
                    mdxnet_input_path, 
                    mdxnet_output_path, 
                    mdxnet_model, 
                    mdxnet_output_format, 
                    mdxnet_overlap, 
                    mdxnet_segment_size, 
                    mdxnet_denoise
                ],
                outputs=[mdxnet_info]
            )
            
            with gr.Row():
                mdxnet_button = gr.Button("Separate!", variant="primary")
            with gr.Row():
                mdxnet_stem1 = gr.Audio(
                    show_download_button=True,
                    interactive=False,
                    label="Stem 1",
                    type="filepath"
                )
                mdxnet_stem2 = gr.Audio(
                    show_download_button=True,
                    interactive=False,
                    label="Stem 2",
                    type="filepath"
                )
            mdxnet_button.click(
                fn=mdxnet_separator, 
                inputs=[
                    mdxnet_audio, 
                    mdxnet_model, 
                    mdxnet_output_format, 
                    mdxnet_segment_size, 
                    mdxnet_overlap, 
                    mdxnet_denoise
                ],
                outputs=[mdxnet_stem1, mdxnet_stem2]
            )
        
        # VR ARCH Tab
        with gr.TabItem("VR ARCH"):
            with gr.Row():
                vrarch_model = gr.Dropdown(
                    label="Select the Model",
                    choices=VRARCH_MODELS,
                    interactive=True
                )
                vrarch_output_format = gr.Dropdown(
                    label="Select the Output Format",
                    choices=OUTPUT_FORMATS,
                    interactive=True
                )
            with gr.Row():
                vrarch_window_size = gr.Dropdown(
                    label="Window Size",
                    choices=VRARCH_WINDOW_SIZES,
                    value=VRARCH_WINDOW_SIZES[0],
                    interactive=True
                )
                vrarch_aggression = gr.Slider(
                    minimum=1,
                    maximum=50,
                    step=1,
                    label="Aggression",
                    info="Intensity of primary stem extraction.",
                    value=5,
                    interactive=True
                )
                vrarch_tta = gr.Checkbox(
                    label="TTA",
                    info="Enable Test-Time-Augmentation; slow but improves quality.",
                    value=True,
                    interactive=True,
                )
                vrarch_high_end_process = gr.Checkbox(
                    label="High End Process",
                    info="Mirror the missing frequency range of the output.",
                    value=False,
                    interactive=True,
                )
            with gr.Row():
                vrarch_audio = gr.Audio(
                    label="Input Audio",
                    type="numpy",
                    interactive=True
                )
            with gr.Accordion("Separation by Link", open=False):
                with gr.Row():
                    vrarch_link = gr.Textbox(
                        label="Link",
                        placeholder="Paste the link here",
                        interactive=True
                    )
                with gr.Row():
                    gr.Markdown("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)")
                with gr.Row():
                    vrarch_download_button = gr.Button(
                        "Download!",
                        variant="primary"
                    )
            vrarch_download_button.click(
                fn=download_audio,
                inputs=[vrarch_link],
                outputs=[vrarch_audio]
            )
            
            with gr.Accordion("Batch Separation", open=False):
                with gr.Row():
                    vrarch_input_path = gr.Textbox(
                        label="Input Path",
                        placeholder="Place the input path here",
                        interactive=True
                    )
                    vrarch_output_path = gr.Textbox(
                        label="Output Path",
                        placeholder="Place the output path here",
                        interactive=True
                    )
                with gr.Row():
                    vrarch_batch_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    vrarch_info = gr.Textbox(
                        label="Output Information",
                        interactive=False
                    )
            vrarch_batch_button.click(
                fn=batch_separator, 
                inputs=[
                    vrarch_input_path, 
                    vrarch_output_path, 
                    vrarch_model, 
                    vrarch_output_format, 
                    vrarch_window_size, 
                    vrarch_aggression, 
                    vrarch_tta, 
                    vrarch_high_end_process
                ],
                outputs=[vrarch_info]
            )
            
            with gr.Row():
                vrarch_button = gr.Button("Separate!", variant="primary")
            with gr.Row():
                vrarch_stem1 = gr.Audio(
                    show_download_button=True,
                    interactive=False,
                    label="Stem 1",
                    type="filepath"
                )
                vrarch_stem2 = gr.Audio(
                    show_download_button=True,
                    interactive=False,
                    label="Stem 2",
                    type="filepath"
                )
            vrarch_button.click(
                fn=vrarch_separator, 
                inputs=[
                    vrarch_audio, 
                    vrarch_model, 
                    vrarch_output_format, 
                    vrarch_window_size, 
                    vrarch_aggression, 
                    vrarch_tta, 
                    vrarch_high_end_process
                ],
                outputs=[vrarch_stem1, vrarch_stem2]
            )
        
        # Demucs Tab
        with gr.TabItem("Demucs"):
            with gr.Row():
                demucs_model = gr.Dropdown(
                    label="Select the Model",
                    choices=DEMUCUS_MODELS,
                    interactive=True
                )
                demucs_output_format = gr.Dropdown(
                    label="Select the Output Format",
                    choices=OUTPUT_FORMATS,
                    interactive=True
                )
            with gr.Row():
                demucs_shifts = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    label="Shifts",
                    info="Number of predictions with random shifts, higher = slower but better quality.",
                    value=2,
                    interactive=True
                )
                demucs_overlap = gr.Dropdown(
                   label="Overlap",
                   choices=DEMUCUS_OVERLAPS,
                   value=DEMUCUS_OVERLAPS[0],
                   interactive=True
                )
            with gr.Row():
                demucs_audio = gr.Audio(
                    label="Input Audio",
                    type="numpy",
                    interactive=True
                )
            with gr.Accordion("Separation by Link", open=False):
                with gr.Row():
                    demucs_link = gr.Textbox(
                        label="Link",
                        placeholder="Paste the link here",
                        interactive=True
                    )
                with gr.Row():
                    gr.Markdown("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)")
                with gr.Row():
                    demucs_download_button = gr.Button(
                        "Download!",
                        variant="primary"
                    )
            demucs_download_button.click(
                fn=download_audio,
                inputs=[demucs_link],
                outputs=[demucs_audio]
            )
            
            with gr.Accordion("Batch Separation", open=False):
                with gr.Row():
                    demucs_input_path = gr.Textbox(
                        label="Input Path",
                        placeholder="Place the input path here",
                        interactive=True
                    )
                    demucs_output_path = gr.Textbox(
                        label="Output Path",
                        placeholder="Place the output path here",
                        interactive=True
                    )
                with gr.Row():
                    demucs_batch_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    demucs_info = gr.Textbox(
                        label="Output Information",
                        interactive=False
                    )
            demucs_batch_button.click(
                fn=batch_separator, 
                inputs=[
                    demucs_input_path, 
                    demucs_output_path, 
                    demucs_model, 
                    demucs_output_format, 
                    demucs_shifts, 
                    demucs_overlap
                ],
                outputs=[demucs_info]
            )
            
            with gr.Row():
                demucs_button = gr.Button("Separate!", variant="primary")
            with gr.Row():
                demucs_stem1 = gr.Audio(
                    show_download_button=True,
                    interactive=False,
                    label="Stem 1",
                    type="filepath"
                )
                demucs_stem2 = gr.Audio(
                    show_download_button=True,
                    interactive=False,
                    label="Stem 2",
                    type="filepath"
                )
            with gr.Row():
                demucs_stem3 = gr.Audio(
                    show_download_button=True,
                    interactive=False,
                    label="Stem 3",
                    type="filepath"
                )
                demucs_stem4 = gr.Audio(
                    show_download_button=True,
                    interactive=False,
                    label="Stem 4",
                    type="filepath"
                )
            demucs_button.click(
                fn=demucs_separator, 
                inputs=[
                    demucs_audio, 
                    demucs_model, 
                    demucs_output_format, 
                    demucs_shifts, 
                    demucs_overlap
                ],
                outputs=[demucs_stem1, demucs_stem2, demucs_stem3, demucs_stem4]
            )
        
        # Themes Tab
        with gr.TabItem("Themes"):
            select_themes()
        
        # Credits Tab
        with gr.TabItem("Credits"):
            gr.Markdown(
                """
                UVR5 UI created by **[Eddycrack864](https://github.com/Eddycrack864).** Join the **[AI HUB](https://discord.gg/aihub)** community.
                * python-audio-separator by [beveradb](https://github.com/beveradb).
                * Special thanks to [Ilaria](https://github.com/TheStingerX) for hosting this space and assistance.
                * Thanks to [Mikus](https://github.com/cappuch) for help with the code.
                * Thanks to [Nick088](https://huggingface.co/Nick088) for assistance in fixing roformers.
                * Thanks to the [yt_dlp](https://github.com/yt-dlp/yt-dlp) developers.
                * Separation by link source code and improvements by [Blane187](https://huggingface.co/Blane187).
                * Thanks to [ArisDev](https://github.com/aris-py) for porting UVR5 UI to Kaggle and making improvements.
                
                You can donate to the original UVR5 project here:
                [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/uvr5)
                """
            )
    
    if __name__ == "__main__":
        parser = ArgumentParser(description="Separate audio into multiple stems")
        parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
        parser.add_argument('--listen-port', type=int, help="The listening port that the server will use.")
        args = parser.parse_args()
        
        app.launch(
            share=args.share_enabled,
            server_name="",
            server_port=args.listen_port or 9999,
        )
