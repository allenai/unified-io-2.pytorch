"""Task-specific prompts"""
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

PROMPT_DICT['video_tagging'] = {
  'original': [
    'What action is being performed in this video? Please provide a short answer.',
    'What activity is going on in the video? Short answer:',
    'Please provide a short answer. What activity is going on in the video?',
  ],
  'manual': [
    'What is the salient activity of the video? Please provide a short answer.',
    'What are they doing in the video with short answer?',
    'What are they doing in this video? Short answer: ',
    'What action does this video depict? Please provide a short answer.',
    'Please provide a short answer. What action takes place in this video?',
    'What is the most prominent action in this video?',
    'Name the most prominent activity in this video with a short answer.',
  ],
  'gpt3': [
    "What is happening in this video clip? Short answer:",
    "Can you describe the activity occurring in this video with a few words?",
    "What activity is shown in this video in short answer?",
    "What is the action demonstrated in this video? Please provide a short answer.",
    "What event is being depicted in this video? Please provide a short answer.",
    "Can you tell what's going on in this video with a few words?",
    "What kind of action does this video portray in short answer?",
    "What is the action happening in this video clip? Please provide a short answer.",
    "What is the specific action depicted in this video with a few words?",
    "What sort of action is the video showing? Short answer:",
    'Could you briefly tell me what\'s happening in this video?',
    'Give a short response: What\'s the main action in the video?',
    'What\'s the main event in the video? Respond briefly, please.',
    'What\'s the central activity in the footage? Please keep the answer short.',
    'Please succinctly describe the action in the video.',
    'In a few words, what is happening in this video?',
    'What is the video showing? Provide a short explanation.',
    'Please tell me in brief: What event is captured in this video?',
    'What\'s the significant action in the footage?',
    'Provide a brief answer about the main event in the video.',
    'What is being carried out in this video? Briefly describe.',
    'In few words, what\'s the primary event in the video?',
    'What does the footage principally depict? A short answer, please.',
    'Could you summarise the main action in the video?',
    'In brief, what\'s the principal event unfolding in this video?',
    'What are the subjects in the video doing? Give a short answer.',
    'What\'s the main occurrence in the video? Please respond briefly.',
    'Please give a succinct account of the video\'s main activity.',
    'Identify the central action in this video in a few words.',
    'What\'s the crucial event in the footage? Respond concisely, please.',
    'Describe the key activity in this video briefly.',
    'In a nutshell, what\'s unfolding in the video?',
    'Please briefly describe the main occurrence in the video.',
    'What\'s the significant event in the video? Provide a succinct response.',
    'Describe in few words, the main action captured in this video.',
    'What action is the video mainly highlighting? Please keep your answer short.',
    'Can you briefly explain the primary activity in this video?',
    'What\'s the primary focus of the video in terms of activity? Respond shortly, please.',
    'In a concise answer, what\'s the video principally showing?',
    'What is the subject doing in the video? Provide a brief response.',
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

PROMPT_DICT['image_generation_coco_2017'] = dict(
  original=['Generate an image matching this text: {}'],
  manual=[
    'What image corresponds to the text description "{}"?',
    'Complete the image corresponding to the caption: "{}"',
    'What image can be described as "{}"?',
    'Construct a picture that could have this caption: {}"?',
    'Text: {}\nGenerate an image that matches the text.',
    'Draw an image that can be described as "{}".',
    'Draw an image that could be captioned "{}".',
    'Generate an image that could be captioned "{}".',
    'Caption: {}\nWhat is an image that matches the caption?',
    'What might \"{}\" look like?',
    'Generate an image of {}',
    'Generate a picture of {}',
    'Scene: {}\nInstruction: Please draw the scene',
  ],
  gpt=[
    'Draw an image that matches the caption "{}"',
    'Illustrate an image that aligns with the description "{}".',
    'Create an image that might be captioned as "{}".',
    'Conceive an image that would suit the caption "{}".',
    'Fabricate an image with a possible caption of "{}".',
    'Render an image that matches {}.',
  ]
)


PROMPT_DICT['Surface_Normals_Estimation'] = dict(
  original=[
    'What is the surface normal of the image ?'
  ],
  manual=[
    'Generate an image depicting the surface normal at each pixel of the input image.',
    'Generate an image representing the orientation of the surface at each pixel.'
  ],
  gpt3=[
    'Create an image that shows the orientation of the surface at each pixel.',
    'Generate an image that indicates the orientation of the surface at each pixel.',
    'Make an image that displays the orientation of the surface at every pixel.',
    'Generate the surface normals of this image.',
    "Could you determine the surface normal of the picture?",
    "Produce the surface normals for this image.",
    "Find the surface normals for this particular image.",
    "Can you ascertain the surface normal for this image?",
    "What would be the surface normal for this picture?",
    "Determine the surface normal of the image.",
    "Could you establish the surface normals for this picture?",
    "What's the image's surface normal?",
    "Make the surface normals for this image.",
    "Could you tell me the surface normal for this image?",
    "Formulate the surface normals for this image.",
    "Can you identify the surface normal in the image?",
    "Compute the surface normals for this image.",
    "What is the normal vector for the image surface?",
    "Construct the surface normals for this given image.",
    "What's the normal direction for the image surface?",
  ]
)


PROMPT_DICT["Detection_Generic"] = dict(
  original=['List all objects and their locations in the image'],
  manual=[
    'Enumerate the objects in the picture. For each one, list its category and bounding box.',
    'Locate all objects in the image.',
    'Detect all of the objects in the image.',
    'Detect all of the objects in <image_input>.',
    'Find all the objects.',
    'Find regions of all of the objects in the image.',
    'There are one or more objects in this image. Find the regions corresponding to each of them.',
    'For each object in the image, return the object category and bounding box',
    "Perform object detection by returning a class and bounding box for each object in this image"
  ]
)


PROMPT_DICT["Detection_COCO"] = dict(
  original=["Find the COCO objects."],
  manual=[
    'List the COCO objects and their locations in the image',
    'Locate all COCO objects in the image.',
    'Detect all of the COCO objects in the image.',
    'Find all the COCO objects.',
    'Find regions of all of the COCO objects in the image.',
    'There are one or more COCO objects in this image. Find the regions corresponding to each of them.',
    'For COCO object in the image, return the object category and bounding box'
  ]
)


PROMPT_DICT['Object_Segmentation'] = dict(
  original=[
    'Segment the object in this bounding box: {}',
  ],
  manual=[
    'Building a binary mask of the pixels that part of the main object in {}',
    'Generate a black and white image. The white pixels should be ones that are part of the object in {}',
    'Segment the object at {}',
    'There is an object at {}. Building a binary mask containing just the pixels of that object',
    "Object segmentation requires building a binary mask with white pixels that are part of a particular object and black pixels"
    " everywhere else. Do object segmentation for the object at {}",
    "Show me what pixels are part of the object at {}",
    "What are the exact pixels that belong to the main object in the bounding box {}?"
  ],
  gpt3=[
    'Select the pixels that match the description " {} ".',
    'Highlight the pixels that fit the description " {} ".',
    'To do object segmentation, you must find the pixels that represent each instance of a category. Find the object segmentation of " {} ".',
    'Identify the pixels that correspond to each instance of the category " {} ".',
    'Mark all the pixels that depict " {} ".',
  ]
)


PROMPT_DICT['Depth_Estimation'] = dict(
  original=[
    'What is the depth map of the image ?',
  ],
  manual=[
    'Assign each pixel in this image a depth value, and generate a grayscale image representing the depth at each pixel.',
    'Determine the depth of each pixel of this image, and generate a map of the same.',
    'Generate a depth map of this image, where darker pixels correspond to less depth.',
    'Depth image: ',
    'Generate the depth image.',
    "What is the depth?",
  ],
  gpt3=[
    "Allocate a depth value to every pixel in the picture and create a grayscale image representing the depth at every pixel.",
    "Ascertain the depth of each pixel in this image and create a corresponding map.",
    "Produce a depth map of the picture where darker pixels signify less depth.",
    "Image demonstrating depth: ",
    "Create an image showing depth.",
    "Could you determine the depth?",
    "Assign a depth value for each pixel in this image and represent it through a grayscale image.",
    "Map the depth of each pixel in this image and generate a corresponding visual.",
    "Form a depth map where the image's darker pixels indicate shallower depth.",
    "Depth portrayal in the image: ",
    "Develop an image that shows the depth.",
    "What's the depth measurement?",
    "Give each pixel a depth value in this image, and form a grayscale image showing the depth of each pixel.",
    "Identify the depth for every pixel in this image and construct a matching map.",
    "Create a depth map for this image, with darker pixels indicating less depth.",
    "Depiction of depth in the image: ",
    "Generate an image that indicates the depth.",
    "Can you provide the depth?",
    "Set a depth value for all pixels in this image and construct a grayscale image to reflect this depth.",
    "Gauge the depth of each pixel in this image and produce a map to show it.",
  ]
)

PROMPT_DICT['image_tagging_imagenet2012'] = {
  "original": ["What is this? Return a precise but very short answer."],
  "manual": [
    'What is the most prominent object in this image? Please provide a short answer.',
    'Name the most prominent object in this image with a short answer.',
    'Give me the name of the object in the image',
    'Give me a fine-grained tag for the thing in the photo',
    'Generate a fine-grained tag for the thing in <image_input>',

    'Please provide a short answer. What does <image_input> contain?',
    "What is this? Give me a precise but very short, few-word answer.",
    "What is the main object in the photo? Return a short but fine-grained category.",
    "Return a fine-grained class that matches the object in the image.",
    "State, precisely and succinctly, what is shown here",
    "Tell me a fine-grained category that matches the object in the image.",

    'Can you identify the object in <image_input>? Give a brief response.',
    'Identify the object in the image. Brief response:',
    'Can you describe briefly what\'s in the image?',
    'What does the image show? Kindly give a concise answer.',
    'In a few words, can you tell what object is depicted in the image?',
    'Could you briefly explain what object is presented in the image?',

    'Identify the main object in this image in a few words',
    'Provide a brief description of the most noticeable object in this image.',
    'What\'s the single object highlighted in this image?',
    'Briefly, what is the primary content of this image?',
    'What object stands out the most in this image? Please give a succinct response.',
    'Name the object that is the most striking in this image briefly.',

    'What is the principal item featured in this image? Keep the answer short.',
    'Offer a concise response: What\'s the main focus of this image?',
    'Please provide a quick answer. What\'s the chief subject of the image?',
    'Can you identify the central subject in this image? A short answer will do.',
    'What\'s the foremost entity captured in this picture? Keep your response brief.',
    'Give a succinct description of the primary thing in this picture.',
    'In this image, what\'s the dominant figure or object? Please provide a short answer.',
    'What\'s the major object in this image? Kindly provide a succinct reply.',
    'Quickly name the primary object featured in this picture.',
    'Categorize the key entity depicted in the image in a few words.',
    'Which object stands out the most? Give a short answer.',
    'In just a few words, what\'s the notable object in this image?',
  ]
}


PROMPT_DICT['Audio_Generation'] = dict(
  original=[
    'Synthesize the sound based on the description "{}"',
  ],
  manual=[
    '{}',
    'Generate the sound based on the description: {}',
    'What might \"{}\" sound like?',
    'What is the sound of {}',
    'Create the noise of: {}',
    "Event: {} Generate a noise this event could plausibly make."
  ],
  gpt3 = [
    "Generate the sound in accordance with the description '{}'.",
    "Create audio as described in '{}'.",
    "Develop a sound that fits the description '{}'.",
    "Produce the sound that matches the description '{}'.",
    "Formulate sound based on the given description '{}'.",
    "Construct a sound representation according to '{}'.",
    "Design a sound as characterized by '{}'.",
    "Generate audio corresponding to the details in '{}'.",
    "Create a sound consistent with the description '{}'.",
    "Develop audio according to the provided description '{}'.",
    "Fabricate a sound as per the given details '{}'.",
    "Create sound that fits the guidelines '{}'.",
    "Formulate a sound following the instructions '{}'.",
    "Craft audio as specified in '{}'.",
    "Devise a sound based on the information '{}'.",
    "Engineer a sound output matching '{}'.",
    "Produce audio that aligns with '{}'.",
    "Render a sound that is consistent with '{}'.",
    "Model a sound that corresponds with the description '{}'.",
    "Generate a sound that adheres to the description '{}'.",
  ]
)


class Prompt:
  """Utility class to access prompts"""

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
