# LLMs-Solving-Word-Puzzles

Note that I have not organized this project yet. I wrote these all as seperait scripts to build my dataset. 

Use `pip install -r ./requirements.txt` for project dependencies. 

The main file you want to run is `build_world_puzzle_sft_jsonl.py` (this is the file I used to build the dataset in hugging face). 

https://huggingface.co/datasets/eli-equals-mc-2/WordPuzzleDataset

Currently, I am working on the getting the SFT model I trained to Hugging face. 

You can take a look at the presentation as well. 

I use the `Open-R1` github repository for SFT training recipes: `https://github.com/huggingface/open-r1`. I used this project with my dataset to train the Qwen-1.5b model. 

More details coming. 