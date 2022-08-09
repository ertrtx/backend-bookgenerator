## Main function called by flask_api.py to generate book (text + images)

import os, json, datetime
from GPTJGenText import generateText
from imagenMod import generateImages

# prompt = "The red fox jumped the fence"

def generateBook(prompt, model, HFtokenizer):

    print("start processing: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    currentDir = os.getcwd()
    secretPath = os.path.join(currentDir, 'gpt-j','secret')

    with open(os.path.join(secretPath,'my_vars.json'), 'r') as f:
        myDict = json.load(f)
        project_id = myDict["project_id"]
        bucket_name = myDict["bucket_name"]
        f.close()

    keyFile = os.path.join(secretPath,'my_keys.json')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= keyFile

    generatedPromptsList = generateText(prompt, currentDir, model, HFtokenizer)
    print(generatedPromptsList)
#    for count, prompt in enumerate(generatedPromptsList):
#            generateImages(prompt, count, currentDir, project_id, bucket_name)

    print("finish processing: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

