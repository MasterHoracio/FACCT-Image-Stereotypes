from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm


# ----- setting on a specific gpu (UniTo)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ------


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if torch.is_tensor(obj):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

print(torch.cuda.get_device_name(0))


NUM_IMAGES = 100
DIR_IMAGES = "Data/images/"
DIR_TESTS = "Data/tests/"
DIR_JAILBREAKS = "Data/jailbreaks/"
DIR_PROMPT_TESTS = "Data/prompts_tests/"
DIR_PROMPT_QUESTION_IMGS = "Data/question_imgs/"

def load_image_questions(dir_path, question_img, processor, type_approach = None):
    image_questions = []
    file_names = [(f.name, f.path)for f in os.scandir(dir_path) if f.is_file()]
    #format_img_question = "<|im_start|>USER: <image> \n{}<|im_end|><|im_start|>ASSISTANT:"
    if type_approach == "zero" or type_approach == "guided" or type_approach is None:
        for (f_name, f_path) in file_names:

            prompt, conversation = get_actual_prompt(question_img, True, processor, f_path)
            image_questions.append({"name": f_name.split(".")[0], 
                        "img": Image.open(f_path),
                        "prompt": prompt,
                        "conversation": conversation
                        })
    elif type_approach == "cot":
        image_questions = [{"name": f_name.split(".")[0],
                            "path": f_path, 
                    "img": Image.open(f_path),
                    "prompt_list": question_img[0],
                    "general_prompt": question_img[1],
                    } for (f_name, f_path) in file_names]
    
    return image_questions

def dump_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4,  ensure_ascii=False, cls=TensorEncoder)

def read_json(file_name):
    data = {}
    with open(file_name) as f:
        data = json.load(f)
    return data

def read_txt(file_name):
    with open(file_name, "r", encoding= "utf-8") as f:
        list_lines = f.readlines()
        list_lines = [l.strip("\n") for l in list_lines]
    return list_lines

def get_actual_prompt(question, image, processor, image_path = None):

    content = []
    content.append({"type": "text", "text": question})
    if image:
        content.append({"type": "image",
                        "image": image_path})

    conversation = [
                    {
                    "role": "user",
                    "content": content,
                    },
                ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt, conversation

def load_tests(dir_path, lang, general):
    field_general = "general" if general else "school"
    file_names = [(f.name, f.path) for f in os.scandir(dir_path) if f.is_file()]
    tests = {
        f_name.split(".")[0]: [question[field_general][lang] for question in read_json(f_path)]
        for (f_name, f_path) in file_names
    }
    return tests

def get_jailbreak(jail_name, lang): 
    return read_json(DIR_JAILBREAKS + jail_name + ".json")[lang]

def get_prompt_test(prompt_name, lang): 
    return read_json(DIR_PROMPT_TESTS + prompt_name + ".json")[lang]

def get_question_img(question_imgs, lang):
    return read_json(DIR_PROMPT_QUESTION_IMGS + question_imgs + ".json")[lang]


def get_prompt_textual_question(jailbreak, prompt_test, tests, processor):
    list_questions = []
    num_answer = 1
    #format_text = "<|im_start|>USER: {}<|im_end|><|im_start|>ASSISTANT:"
    for (name_test, test_questions) in tests.items():
        for (i, question) in enumerate(test_questions):
            to_ask = prompt_test + " " + question + " " + jailbreak
            prompt, conversation =  get_actual_prompt(to_ask, False, processor)
            question_dict = {"test": name_test,
                             "question": question,
                             "idx_in_test": i,
                             "prompt": prompt,
                             "conversation": conversation}
            list_questions.append(question_dict)
            num_answer += 1

    return list_questions

def get_only_new_tokens(generated_string):
    if ("ASSISTANT:" in  generated_string):
        return generated_string.split("ASSISTANT:")[1]
    else:
        return generated_string

def get_str_date():
    return f"{datetime.now().strftime('%d-%m-%Y %H-%M-%S')}"

def extract_answer(st1, st2):
    idx = st1.find(st2)
    if idx == -1:
        return ""  # st2 no está en st1
    return st1[idx + len(st2):]

def save_args(args, timestamp_now):
    path_file_args = os.path.join(args.model_path, "args_" + timestamp_now + ".json")
    dump_json(vars(args), path_file_args)

def get_response(processor, model, prompt, tokens, images):
    inputs_question = processor(
                    text=[prompt], 
                    images = images,
                    padding = True, 
                    return_tensors="pt"
                    ).to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs_question, 
                                    max_new_tokens=tokens, 
                                    do_sample=False
                                    )
    generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_question.input_ids, generated_ids)
        ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )
    
    return output_text

def run_generation(lang, questions, processor, model, timestamp_now, model_path, type_approach, is_images = False, stereotype_list = None):
    
    for question in tqdm(questions, desc="Processing questions"):
        if type_approach == "cot" and is_images: 
            if lang == "it":
                start = "Gli stereotipi sono delle credenze che possono influenzare la percezione generale della società. Ecco un elenco di potenziali stereotipi connessi a caratteristiche e identità umane:"
            else:
                start = "Stereotypes are beliefs that can influence society's general perception. Here is a list of potential stereotypes related to human traits and identity:"
            full_question_img = start + stereotype_list + "\n" + question["general_prompt"]
            #format_img_question = "<|im_start|>USER: <image> \n{}<|im_end|><|im_start|>ASSISTANT:"
            #prompt_question = format_img_question.format(full_question_img)
            prompt_question, conversation = get_actual_prompt(full_question_img, True, processor, question["path"])
            question["prompt_cot"] = prompt_question 
        else:
            prompt_question = question["prompt"]
            conversation = question["conversation"]
        images = question["img"] if is_images else None

        answer_not_extracted = get_response(processor, model, prompt_question, 256, images)[0]
        question["answer_not_extracted"] = answer_not_extracted
        #decoded = extract_answer(answer_not_extracted, "ASSISTANT:")
        question["answer"] = [answer_not_extracted]
        
    fixed_part_type_generation = "images" if is_images else "textual"
    
    if is_images:
        file_name =  "LlavaNext_" + type_approach + "_" + lang + "_" + fixed_part_type_generation + "_" + timestamp_now + ".json"
    else:
        file_name =  "LlavaNext_" + lang + "_" + fixed_part_type_generation + "_" + timestamp_now + ".json"
    
    path_file = os.path.join(model_path, file_name)

    if is_images: #we exclude the PIL image from the dump
        questions = [{k:v for (k,v) in question_img.items() if k!= "img"} for question_img in questions]

    dump_json(questions, path_file)



def main(args):
    # Load the model and processor
    #model_id = "llava-hf/llava-1.5-7b-hf"  # or use 13B if available
    timestamp_now = get_str_date()
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    save_args(args, timestamp_now)
    
    #TODO make a file with the configuration used (the args) for the experiment
    lang = args.lang
    type_approach = args.type_approach
    jailbreak = get_jailbreak(args.jailbreak, lang)
    prompt_test = get_prompt_test(args.prompts_tests, lang)
    tests = load_tests(DIR_TESTS, lang, args.general)
    """if type_approach == "zero":
        question_img = get_question_img("1", lang)
    elif type_approach == "cot":
        question_img = get_question_img("3", lang)
    elif type_approach == "guided":
        question_img = get_question_img("2.1", lang)"""
    question_img = get_question_img(args.question_imgs, lang)

    model_id = args.model_id
    processor = LlavaNextProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    processor.patch_size = 14 #got this and patch_size from https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/main/processor_config.json
    processor.vision_feature_select_strategy = "default"
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    #we ask the llm the various textual questions from the questionaire first
    textual_questions = get_prompt_textual_question(jailbreak, prompt_test, tests, processor)
    run_generation(lang, textual_questions, processor, model, timestamp_now, args.model_path, type_approach, False)

    #we now ask to rate the images one by one
    images_questions = load_image_questions(DIR_IMAGES, question_img, processor, type_approach)
    if type_approach == "cot":
        #format_text = "<|im_start|>USER {}<|im_end|><|im_start|>ASSISTANT:"
        #prompt = format_text.format(images_questions[0]["prompt_list"])
        prompt, conversation = get_actual_prompt(images_questions[0]["prompt_list"], False, processor) 
        answer_stereotype_list = get_response(processor, model, prompt, 512, None)[0]
        print(f"answer_st: {answer_stereotype_list}")
        #stereotype_list = extract_answer(answer_stereotype_list, "ASSISTANT:")
        #print(f"stereotype_list: {stereotype_list}")
        run_generation(lang, images_questions, processor, model, timestamp_now, args.model_path, type_approach, True, answer_stereotype_list)
    elif type_approach == "zero" or type_approach == "guided":
        run_generation(lang, images_questions, processor, model, timestamp_now, args.model_path, type_approach, True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True) #where we save the results
    parser.add_argument("--model_id", type=str, required=True) #id of the model to use
    parser.add_argument("--lang", type=str, required=True, choices=["it", "en"]) #language of the prompts and tests
    parser.add_argument("--type_approach", type=str, required=True, choices=["zero", "cot", "guided"]) # type of prompting approach (for detecting stereotypes in images)
    parser.add_argument("--general", type=bool, default=True) #if we are asking the question in a general manner or targeted towards teachers
    parser.add_argument("--jailbreak", type=str, default="1") #what jailbreak file (in the folder Data/jailbreaks) we use, you have to only pass the name, the .json extesion is added by default
    parser.add_argument("--question_imgs", type=str, default="1") #what questions for img file (in the folder Data/question_imgs) we use, you have to only pass the name, the .json extesion is added by default
    parser.add_argument("--prompts_tests", type=str, default="1") #what question file (in the folder Data/prompts_tests) we use, you have to only pass the name, the .json extesion is added by default
    parser.add_argument("--max_new_tokens", type=int, default=128) #max new tokens generated
    parser.add_argument("--max_new_tokens_img", type=int, default=256) #max new tokens generated for the images
    args = parser.parse_args()
    print(args)
    main(args)