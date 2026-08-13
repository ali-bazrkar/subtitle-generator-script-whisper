# subtitle-generator-script-whisper
its a small script i have vibe coded for generating subtitle for videos i download online
it uses whisper models to generate subs

you probably would want to use UV just like i did or simply download the dependencies manually from the .py files using pip if you specifically dont want to use uv.

another thing you might want to do is to use a Virtual environment (``` uv env ```) before running anything so when you run the scripts all dependencies will stay within your project folder but thats really up to you.

the script is simple, loads the mode, finds the cuda related files, and looks for video formats inside the directory you pass to it.
if you want you could use CPU mode, simply change the line ``` model = WhisperModel(model_path, device="cuda", compute_type="int8") ```
device to ```"cpu"``` instead of ```"cuda"```. also ```compute_type``` can be changed too. the higher it is the more accurate it works but also uses more resources and probably more heat. depends on the code but my gpu only supported int8 so i kept it as is. 

the ```model_path``` is pointing to the local folder (in models) where i keep my downloaded models locally there. you can download them too if you want, as many as you wish
if you have not downloaded it yourself, the script will point to the appropriate HF repo and it will be downloaded as you just run the script but it probably will be stored where uv / whisper decides to keep it
i didnt really check because i prefer to have them all in one folder so i download/clone them manually and just point to them so i know where everything related to my project lives.

if you want to get them locally

> ``` git clone https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2 [DESTINATIO_FOLDER] ```

> ``` git clone https://huggingface.co/Systran/faster-whisper-small.en [DESTINATIO_FOLDER] ```

they are the repos i use to pull the models from
to run the script you simply ``cd`` into the project 

and run 
> ``` uv run subgen.py <destination>```

destination is where your videos live and you aim to generate subs for them

the normal ``subgen.py`` uses **"turbo"** model
if you want a lighter model i use **"small-en"**
the "-en" series are specifically designed for english subtitle only.
they have higher accuracy compared to the multilingual series so if your video contains non-english stuff you might wanna try other models 
check the official openAI whisper repo you will see what models they offer and what the differences are there.

generally **turbo** is very efficient and fast and accurate but still heavier

**small-en** is very light. i also used to generate subs on **small-en** and **base-en** on CPU before. thats an overkill i would not recommend it unless you have to because i couldnt get the cuda work i used CPU but current script will load the needed cuda files after you run the python script (and uv automatically fetches and downloads the dependencies)

> ``` uv run subgen-small-en.py <destination>```

the only differences is really the model used, i made them separate so i can simply switch in the terminal with whichever model i want but keep in mind that the not specified py file is **turbo** model

one other thing is worth to note, after each video is generated a srt file, it will wait for a certain time:
for turbo (main script) i set it to 20 seconds rest per video
for small-en its 8 second. 

the small-en even without the rest stabilizes around 80 C on my gpu, so the timer was not really needed
but the turbo keeps growing hotter without the rest so its a good idea to tune it with your own GPU if needed.
a rest in between of videos if you have lots of videos to transcribe can prevent overheating 
