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

PROMPT_DICT['Box_Classification_Scene'] = {
  "original": ['What is the category of the object at {box}?'],
  "manual": [
    'What is the category item in the region {box}?',
    'What is the category of the object in the region " {box} "?',
    'What kind of object is in the region {box} of <image_input>?',
    'What type object is located at " {box} "?',
    'State the type object that is located in this bounding box: {box}.',
    'There is an object at {box}, please categorize it.',
    'Categorize {box}',
    'Tag {box} in <image_input>',
    'Tag the object at {box}.',
    'Instruction: Categorize the object\nContext: {box}\n',
    'Name the object in {box}',
    'Name the object in {box} of the photograph',
    'Help me figure out what kind of object is located at {box} in <image_input>',
    'Region: {box}\nInstruction: Name the object in the region\n',
    'Very briefly, tell me what object is in " {box} ".',
  ],
  "gpt3": [
    'Can you identify the type of object that is located at {box}?',
    'Name the object in the region: {box}',
    'What type of object is present in the area " {box} "?',
    'What is the object situated in the " {box} " region? Respond with just a few words.',
    'Which object is positioned at " {box} "?',
    'An object exists at {box}, can you identify it?',
    'What category of item is located in the zone " {box} "?',
    'What is the item in the " {box} " area?',
    'There\'s something at {box}, can you tell what it is?',
    'What kind of object can be found in {box}?',
    'Identify the object in the region {box}.',
    'Can you specify the object\'s category in the area {box}?',
    'Which object is present in the " {box} " region?',
    'An object is spotted at {box}, can you specify it?',
    'What class of object lies in the part " {box} "?',
    'What is the object in " {box} " area?',
    'Which object resides in " {box} "?',
    'Something is located at {box}, what would you categorize it as?',

    'What kind of object can be found in the section {box}?',
    'Identify the object in the region {box}.',
    'Can you specify the object\'s category in the area {box}?',
    'What class of object lies in the part {box}?',
    'Briefly describe the object in {box} area?',
    'Give a few word description of the object that resides in {box}?',
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

PROMPT_DICT['Object_Detection_No_Cat'] = {
  "original": ['Which regions does the text "{}" describe?'],
  "manual": [
    "Objects: {}\nInstruction: Return the location of the objects in the image",
    "I am looking for {} in this picture, can you show me where they are?",
    "What regions contain {} in <image_input>?",
    'Find image regions that match: {}',
    'Find all instances of {}.',
    'Please find all instances of {} in <image_input>',
    'Locate all instances of {}.',
    'List the regions of the image that match {}.',
    'Find all regions in the image corresponding to the category "{}".',
  ],
  "gpt3": [
    'Find every "{}" in this image.',
    'Please locate all of the {} in this image.',
    'Search the image for all regions corresponding to the text "{}".',
    'Identify all "{}" regions in the image.',
  ]
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


PROMPT_DICT['Pose_Estimation'] = {
  'original': [
    'Find the visible human joints in the region " {} ".'
  ],
  'manual': [
    'Find the visible keypoints corresponding to the person located at {}',
    'Return the visible keypoints on the human body of the person in the region {}',
    'Find the visible human joints of the person located at {}.',
  ],
  'gpt3': [
    'Locate the visible human joints in the region {}.',
    'Where are the visible human joints in the region {} ?',
    'Could you please find the visible joints of a human in the region {}.',
    'Locate the joints of the person located at {}.',
    'Identify the visible joints of the human located at {}.',
    'Find the visible keypoints on the human body in the region {}.',
    'Identify the visible keypoints on the human body appearing at {}.',
    "Identify the noticeable keypoints for the person found at {}.",
    "Spot the visible human joints of the individual positioned at {}.",
    "Detect the discernible keypoints associated with the person at location {}.",
    "Locate the observable keypoints for the person situated at {}.",
    "Point out the visible joints on the human body located at {}.",
    "Track the observable joints of the person in the region {}.",
    "Spot the discernible human joints located in the region {}.",
    "Identify the apparent keypoints for the individual within the region {}.",
    "Find the keypoints for the person stationed at {}.",
    "Locate the discernible joints of the person present at {}.",
    "Detect the human joints in the {} region that are visible.",
    "Track the person's visible keypoints found in the region {}.",
    "Trace the visible keypoints of the person located in {}.",
  ]
}


PROMPT_DICT['video_captioning'] = {
  "original": [
    'What does the video describe ?',
    'Caption the video with a short description.',
    'Write a short description of this video.',
    'Describe the video with a short description.',
    'Write a brief caption describing the video.'
  ],
  'gpt3': [
    "What is the video portraying in a short description? ",
    "Can you tell what the video is depicting in short?",
    "Can you briefly explain what is the video illustrating ?",

    'Give a brief summary of the video.',
    'In a short statement, caption the video.',
    'What is the video showing? Please describe in short.',
    'Describe the content of the video briefly.',
    'Please summarize the video in a few words.',
    'What is the main theme of the video? Please write briefly.',
    'What\'s the video about? Give a concise description.',
    'Briefly, what does this video present?',
    'In brief terms, explain the content of the video.',
    'Give a succinct summary of what the video shows.',
    'In a few words, describe the video.',
    'Can you briefly caption this video?',
    'Please give a quick summary of the video.',
    'What\'s the key takeaway from the video? Briefly describe.',
    'Briefly, what is this video all about?',
    'What is the essence of the video? Describe in a sentence.',
    'Write a short caption that summarizes the video.',
    'Please provide a brief caption for the video.',
    'In short, what does this video depict?',
    'How would you briefly summarize the video?',
    'What\'s the brief overview of the video?',
    'Provide a concise narrative of the video.',
    'Can you encapsulate the video\'s content in a few words?',
    'Briefly capture the essence of the video.',
    'Give a short explanation of what the video is about.',
    'What\'s the main point of the video? Provide a brief description.',
    'In a nutshell, what is the video presenting?',
    'What\'s the short summary of the video?',
    'Write a brief description encapsulating the video.',
  ]
}

PROMPT_DICT['audio_caption']['original'] = [
  'Give a short description of this audio.',
  'Caption the audio with a short description.',
  'Write a short description of this audio.',
  'Describe the audio in short.',
  'Write a short caption describing this noise.',
  'In brief, what is this sound?'
]


class Prompt:
  def __init__(self, original_flag=True, manual_flag=True,
               gpt3_flag=True, single_prompt=False):
    self.prompt_list = []
    self.original_flag = original_flag
    self.manual_flag = manual_flag
    self.gpt3_flag = gpt3_flag
    self.single_prompt = single_prompt

  def random_prompt(self, task_name, dataset_name=None):
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
    if self.single_prompt:
      return prompt_list[0]
    ix = np.random.randint(0, len(prompt_list))
    return prompt_list[ix]
