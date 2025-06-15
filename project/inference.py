from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "./trainer_output/checkpoint-1000"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model     = AutoModelForCausalLM.from_pretrained(checkpoint).cuda()
system_prompt = "You are a word-formation agent for a Bananagrams-style game. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </thinkz> and <answer> </answer> tags."
user_prompt   = "Available letters (each can be used once): L P P A R D I O V E N T C N A L E T E Y D I O E I A C N T L A E N I A K I N E N L D E G D N G P I I O N Y T C R U I L M Y I G T A H R C E S H F S K H I E I K I R D S U T N H E R E R A R G A A Y A A D M E R N B U N I U M L I H T N \n\nForm as many valid English words as you can using these letters. Return your answer as a JSON array of uppercase words, e.g.: [\"CAT\", \"BAT\"]."

chatml = (
    "<|im_start|>system\n"
    f"{system_prompt}\n"
    "<|im_end|><|im_start|>user\n"
    f"{user_prompt}\n"
    "<|im_end|><|im_start|>assistant"
)

inputs = tokenizer(chatml, return_tensors="pt").to("cuda")
in_len = inputs["input_ids"].shape[1]
print("prompt tokens:", in_len, 
      "max:", model.config.max_position_embeddings)

# 3) Generate with explicit EOS/PAD
out_ids = model.generate(
    **inputs,
    max_new_tokens=20000,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=1.0,
    top_p=0.9,
)

# 4) Decode only the newly generated portion
generated = out_ids[0, in_len:].tolist()
print(tokenizer.decode(generated, skip_special_tokens=True))