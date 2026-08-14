# subtitle-generator-script-whisper
this repo contains a small vibe coded script for generating subtitles for videos i download across the internet.
this was meant to remain a one-time script for my own personal uses however i decided to upload it here so i could access it anytime on any device later and also share it with my friends when needed. i added interactive mode and I'll probably continue polishing and add stuff i find interesting over time. 

## try-out:

### using UV:
clone the repository and type:
```
uv run subgen.py -i
```
the ``-i`` flag will take you to the interactive mode where you exactly choose what you intend to do step by step.

### using makefile:
you can use the command below to use the makefile **(Linux/macOS)**:
```
make
```
or alternatively:
```
make interactive
```
both commands do equally same thing, they will first check UV's existence in your system path and if you didn't have UV installed, they would create a python VENV (Virtual Environment), install the dependencies and run the command in the interactive mode. if your system had uv it would simply run the first command above and uv will handle everything else. if you are on Windows you might wanna install UV or install the dependencies manually to run the script properly.


### Running without interactive CLI:
to run it directly from command line without interactive CLI:
```
make help
```
will give you explanations about flags you can use and use:
```
make run ARGS="<dir> --flag1 value1 -f2 value2"
```
or alternatively without make:
```
subgen.py [-h] [-m MODEL] [-d {cuda,cpu}] [-i] [-c COOLDOWN] [-l LANGUAGE] [-t {transcribe,translate}] [--compute-type COMPUTE_TYPE] [target_dir]
```
to run the program with your custom settings directly without interactive CLI



## how-does-it-work?
as mentioned under the hood it will use [faster-whisper](https://github.com/SYSTRAN/faster-whisper) models, which is a reimplementation of [Whisper models](https://github.com/openai/whisper) with CTranslate2 format. i never used original whisper models myself but these repackages seem to have lower binary size compared to the original models and they are faster. in terms of quality i have used turbo (on GPU), base-en and small-en (on CPU). base-en was alright, small-en had a very nice accuracy even on technical videos.

#### model-details:
for more detail check the [OG whisper](https://github.com/openai/) repo but i will shortly explain them here too anyway.
there are two variants to these models:
* tiny.en
* base.en
* small.en
* medium.en

these are English-Only models if you already know you are not gonna work on other languages these models often show better accuracy compared to their multilingual variants (large and turbo still offer better accuracy even on english since they are much larger)
* tiny
* base
* small
* medium
* large-v1
* large-v2
* turbo
* large-v3

these are multilingual models, the default model used in whisper itself is **turbo** which is a more efficient and faster implementation of **large-v3**. 
there is an important limitation you need to consider on **turbo**. basically models can do:
- **Transcribing**
- **Translation**

transcribing works as ``Language A audio -> Language A SRT `` while translation works as ``Any Language audio -> English SRT``. it is important to keep in mind **while __turbo__ can do transcribing very well, for translation you cannot use turbo model. you have to choose other multilingual models.** and you do not want to use english-only models on non-english contents obviously :)

## script-details:

### Flags Involved:


> ``-m, --model MODEL``

the model name you wish to use for your task, you can also pass a custom Hugging Face path but the model there must be built with CTranslate2 format


> ``-d, --device {cuda,cpu}``

preforming operation using CPU or CUDA (GPU)


> ``-t, --task {transcribe,translate}``

as discussed about, you can generate subtitles from any supported language to the same language (transcribe) or from any supported language to English (translating), translation only generates english SRT files


> ``-l, --language LANGUAGE``

use language codes like "fa" or "en", by default it will pass "auto" and check the language in runtime. if model was inappropriate for your task/language you will be offered appropriate models interactively


> ``-t, ----compute-type COMPUTE_TYPE``

``int8``, ``float16``, ``float32``, ``int8_float16`` can be selected, i personally use int8. other options may not be supported by the model or your GPU you gotta test it yourself.
> ``-c, --cooldown COOLDOWN``

my script uses **exponential backoffs**. it has a rigid system cooldown combined with dynamically scaling cooldown system. the number you enter (in seconds, say 60 seconds) will be used to give your GPU / CPU rest between each 2 video when generating subtitles. this helps to reduce heat and ensure your device won't overheat due to constantly operating on multiple video files in the same folder. your folder may contain lots of 2 minutes videos, it waits (12 seconds in our example) since the video was shorter. however any video higher than 10 minutes length will use 100% of your COOLDOWN and any small scale will at least have 5 seconds wait with respect to your choice (meaning if you explicitly enter 0 or something lower than 5, it will bypass the minimum and use your preference)


> ``-i, --interactive ``

interactive mode prepares the script for you by asking you questions, it handles errors so you would not mistype, you can always hit "enter" if you want to use default settings.


## what-script-provides:


✅ handles user inputs avoiding mismatches in both interactive and normal modes.

✅ checks for correct path in your system and video existence in the entered path.

✅ based on task/language handles appropriate models in both interactive and normal modes.

✅ supports manually downloaded models in "models" folder, uses Hugging Face if needed.

✅ supports cooldown for SRT Generation between two videos (with exponential back-offs).

✅ offers real-time progress bar on model downloading and SRT generation using **TQDM**.

✅ preloads cuda libs when running on GPU preventing errors related to missing libs.

✅ skips videos with existing SRTs and removes failed or stopped (``ctrl + C``) SRTs

✅ provides a friendly and interactive CLI interface for users to get their works done.


## adding-manual-models

by default you can pass any HF link you want as a flag in non-interactive mode. i personally want the models to be accessed in the same folder i am running the script so i can manage them better (rather than going to find them in cache folders) so i store models in the ``models`` folder in the same project directory. make a folder like ``turbo`` or any name you would like (you can look at **PRESETS**). clone the model repo and pass the folder name inside **models** folder as your model flag and the script will check your directory first. if it fails it will look for it in Hugging Face.

> **Note**: If you manually download a model, make sure it is in **CTranslate2 format**. faster-whisper does not directly load the original PyTorch/Transformers model files they must first be converted to CTranslate2 format.


## screenshots:

<p align="center">
  <img src="https://github.com/user-attachments/assets/a55890d1-6fbf-4285-94f9-e8f5e7fd6d82" width="300">
  <img src="https://github.com/user-attachments/assets/b21c232b-c25d-4159-9dfe-bd4957c71843" width="300">
  <img src="https://github.com/user-attachments/assets/c5e1c468-16bc-4060-ae61-de34d4ccb365" width="300">
</p>

these screenshots give you an overview on how the interactive mode looks like as you run it and how the script will log you information and show realtime progress on generating subtitles (3rd image)
