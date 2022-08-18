## Main function called by flask_api.py to generate book (text + images)

import os, json, datetime
from GPTJGenText import generateText
from imagenMod import generateImages
from google.cloud import storage

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

    generatedPromptsList, generatedNounPromptsList = generateText(prompt, currentDir, model, HFtokenizer)


    print(generatedPromptsList,generatedNounPromptsList)



    def deleteBlobs(bucket_name):
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name)

        for blob in blobs
            blob.delete()

    deleteBlobs(bucket_name)

    for count, prompt in enumerate(generatedNounPromptsList):
        text = generatedPromptsList[count]
        generateImages(prompt, text, count, currentDir, project_id, bucket_name)

    print("finish processing: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
