from collections import defaultdict
import numpy as np

PROMPT_DICT = defaultdict(lambda: dict(original=[], manual=[], gpt3=[], template=[]))


PROMPT_DICT['Refexp'] = {
  "original": ['Which region does the text "{}" describe?'],
  "manual": [
    "Expression: {}\nInstruction: Return the region in the image that matches the expression",
    "What region does \"{}\" match?",
    "What region would a person describe as: {}",
    "Generate a bounding box around {}",
    "Find a region in <image_input> that contains {}",
    "Find me the image region that contains {}",
    "Show me the region in <image_input> that best matches \"{}\"",
    "What region in the picture contains {}?",
    "Report a bounding box that encompasses {}",
    "For the expression \'{}\', what image region matches it?",
    "Identify the image region that best matches the expression: {}",
    "Help me find a particular region in the image.\nDescription: {}\nRegion:",
    "I want to find the {}, can you tell me what region it is in?",
  ]
}

PROMPT_DICT['Object_Detection'] = {
  "original": ['Return the bounding boxes and categories of region matching "{}"'],
  "manual": [
    'Which regions can be described as "{}"? Return their bounding boxes and object categories.',
    'Return the bounding boxes and category names of instances of \"{}\" in the image',
    'Find instances of {} in the <image_input> and return their bounding boxes and category names',
    'Categories: {}\nInstruction: Find instances of the given categories in the image and return their bounding boxes and category names',
    'Find regions containing \"{}\", label each region with its category.',
    'Find the {}, label each region with its category.',
    'Report the region and categories of {} in the image.',
    'Find instances of \"{}\", return the y1 x1 y2 x2 coordinates and category of each instance.',
  ],
}

PROMPT_DICT['Object_Segmentation'] = {
  'original': [
    'Segment the object in this bounding box: {}',
  ],
  'manual': [
    'Building a binary mask of the pixels that part of the main object in {}',
    'Generate a black and white image. The white pixels should be ones that are part of the object in {}',
    'Segment the object at {}',
    'There is an object at {}. Building a binary mask containing just the pixels of that object',
    "Object segmentation requires building a binary mask with white pixels that are part of a particular object and black pixels"
    " everywhere else. Do object segmentation for the object at {}",
    "Show me what pixels are part of the object at {}",
    "What are the exact pixels that belong to the main object in the bounding box {}?"
  ],
  'gpt3': [
    'Select the pixels that match the description " {} ".',
    'Highlight the pixels that fit the description " {} ".',
    'To do object segmentation, you must find the pixels that represent each instance of a category. Find the object segmentation of " {} ".',
    'Identify the pixels that correspond to each instance of the category " {} ".',
    'Mark all the pixels that depict " {} ".',
  ]
}

PROMPT_DICT['VQA_short_prompt']["manual"] = [
  "{} A short answer to the question is",
  "{} Short answer:",
  "Answer this question very succinctly: {}",
  "Please provide a short answer. {}",
  "Look at <image_input> and then give a brief answer to this question: {}",
  "Question: {}\nInstruction: Write a short answer to the question using <image_input>\nAnswer:",
  "Give a very brief answer. {}",
  "Given the image, answer the following question with no more than three words. {}",
  "Use the provided image to answer the question: {} Keep your answer as short as possible:",
  'The question "{}" can be answered using the image. A short answer is',
  '{} Answer the question as briefly as possible.',
  'The short answer of the question "{}" is:',
  "{} Answer the question with a single word or phrase",
]

PROMPT_DICT['image_caption_coco_2017'] = {
  "original": ['Caption this image.'],
  "manual": [
    'Write a caption',
    'Write a caption for <image_input>',
    'Please create a caption',
    'Write a caption for this picture.',
    'Briefly describe this picture.',
    'Can you write a caption for this image?',

    'Caption this image with short description.',
    'Caption <image_input> with one or two sentences.',
    'Generate a caption describing this scene',
    'Provide a one to two sentence high-level summary of this photograph.',
    'What is this picture of? Answer with a short sentence.',
    'Write a short caption of <image_input>',
    'Write a short description of this image.',
    'Write a brief caption describing the image.',
    'A short image caption:',
    'Write a short description for the image.',
    'Briefly describe <image_input>.',
    'Concisely describe this image.',
    'Brief summary of the image:',
  ]
}


class Prompt:
  def __init__(self, original_flag=True, manual_flag=True,
               gpt3_flag=True, single_prompt=False):
    self.prompt_list = []
    self.original_flag = original_flag
    self.manual_flag = manual_flag
    self.gpt3_flag = gpt3_flag
    self.single_prompt = single_prompt

  def random_prompt(self, task_name, dataset_name):
    prompt_list = []
    if self.original_flag:
      prompt_list += PROMPT_DICT[task_name]['original']
    if self.manual_flag:
      if 'manual' in PROMPT_DICT[task_name]:
        prompt_list += PROMPT_DICT[task_name]['manual']
      if 'manual' in PROMPT_DICT[dataset_name]:
        prompt_list += PROMPT_DICT[dataset_name]['manual']
    if self.gpt3_flag:
      if 'gpt3' in PROMPT_DICT[task_name]:
        prompt_list += PROMPT_DICT[task_name]['gpt3']

      if 'gpt3' in PROMPT_DICT[dataset_name]:
        prompt_list += PROMPT_DICT[dataset_name]['gpt3']
    if not prompt_list:
      raise ValueError(f"No prompts for {task_name}/{dataset_name}")
    ix = np.random.randint(0, len(prompt_list))
    return prompt_list[ix]
