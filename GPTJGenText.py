## Based on the work of @mallorbc: https://github.com/mallorbc/gpt-j-6b
## Generate original text from prompt using GPT-J
## https://huggingface.co/docs/transformers/model_doc/gptj

import torch as th
from transformers import GPTJForCausalLM, AutoTokenizer
import datetime
import os
import spacy
from google.cloud import storage
import argparse
import random
import json

import nltk 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize


def loadGPTJModelPlusTokenizer(modelPath):
    if th.cuda.is_available():
        print("CUDA available")
    else:
        print("Check NVIDIA drivers and Pytorch, CUDA not available")
    print("started loading model: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    model =  GPTJForCausalLM.from_pretrained(modelPath, torch_dtype=th.float16).cuda()
    print("finished loading model: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    HFtokenizer = AutoTokenizer.from_pretrained(modelPath)
    model.eval()
    print('done loading tokenizer, .eval() on')
    return model, HFtokenizer


def executeTextInference(prompt, model, HFtokenizer):
    input_ids = HFtokenizer.encode(str(prompt), return_tensors='pt').cuda()
    print("started inference: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    output = model.generate(input_ids, do_sample=True, 
                            min_length=200, max_length=350,top_k=30,top_p=0.7,
                            temperature=0.7,repetition_penalty=2.0,num_return_sequences=3)
    print("finished inference: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return output


def parseSentences(output, HFtokenizer):
    nlp = spacy.load('en_core_web_sm')
    goodOutputs = []
    for i in range(0,len(output)):
        tokens = nlp((HFtokenizer.decode(output[i], skip_special_tokens=True)))
        num_sents = len([sent for sent in tokens.sents])
        sentences = [sent for sent in tokens.sents]
    if num_sents >= 10:
        goodOutputs.append(sentences)
        print('output #{0}: {1} sentences'.format(i, num_sents))
    return goodOutputs


def uploadBlob(project_id, bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client(project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

# Function to extract the nouns 
def colorNounExtractor(sent):
        colorNounPrompts = []
        color_list = ['pink', 'blue', 'yellow', 'rainbow', 'red', 'green', 'purple','magenta', 'cyan', 'brown']
        color = random.choice(color_list)
       # sent = str(sent)
        print(sent)
        words = nltk.word_tokenize(sent)
        # words = [word for word in words if word not in set(stopwords.words('english'))]
        # print(words)
        tagged = nltk.pos_tag(words)
        l = []
        for (word, tag) in tagged:
            # print(word, tag)
            if tag in {'NN','NNP'}: # If the word is a noun
                textGenPrompt = color + ' ' + word
                l.append(textGenPrompt)
        if len(l) == 0:
          l.append(color + ' ' + random.choice(words))
        textGenPromptOut = random.choice(l)
        return textGenPromptOut


def saveAndPassToImageGen(goodOutputs, currentDir):
    pick = random.randrange(len(goodOutputs))
    ii = 0
    generatedTextList = []
    generatedNounPromptList = []
    for i in goodOutputs[pick]:
        if ii > 9:
            break
        filenameLocal = os.path.join(currentDir,'gpt-j','outputs','text_{0}.txt'.format(ii))
        filenameBucket = 'text_{0}.txt'.format(ii)
        myfile = open(filenameLocal, 'w')
        myfile.write(str(i))
        myfile.close()
        generatedTextList.append(str(i))
        # uploadBlob(project_id, bucket_name, filenameLocal, filenameBucket)
        ii+=1
        generatedNounPromptList.append(colorNounExtractor(str(i)))
    return generatedTextList, generatedNounPromptList
    

def generateText(prompt, currentDir, model, HFtokenizer):
    modelPath = os.path.join(currentDir, 'gpt-j', 'model')
    sessionStamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # model, HFtokenizer = loadGPTJModelPlusTokenizer(modelPath)

    numSentsOut = 0
    while numSentsOut < 1:
        output = executeTextInference(prompt, model, HFtokenizer)
        goodOutputs = parseSentences(output, HFtokenizer)
        numSentsOut = len(goodOutputs)
        if numSentsOut >= 1:
            break
    
    generatedPromptsList, generatedNounPromptsList = saveAndPassToImageGen(goodOutputs, currentDir)

    return generatedPromptsList, generatedNounPromptsList

def setupModel():
    currentDir = os.getcwd()
    modelPath = os.path.join(currentDir, 'gpt-j', 'model')
    model, HFtokenizer = loadGPTJModelPlusTokenizer(modelPath)
    return model, HFtokenizer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt')
    args = parser.parse_args()

    currentDir = os.getcwd()
    secretPath = os.path.join(currentDir, 'gpt-j','secret')
    modelPath = os.path.join(currentDir, 'gpt-j', 'model')

    prompt = args.prompt

    with open(os.path.join(secretPath,'my_vars.json'), 'r') as f:
        myDict = json.load(f)
        project_id = myDict["project_id"]
        bucket_name = myDict["bucket_name"]
        f.close()

    keyFile = os.path.join(secretPath,'my_keys.json')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= keyFile

    model, HFtokenizer = loadGPTJModelPlusTokenizer(modelPath)

    numSentsOut = 0
    while numSentsOut < 1:
        output = executeTextInference(prompt, model, HFtokenizer)
        goodOutputs = parseSentences(output, HFtokenizer)
        numSentsOut = len(goodOutputs)
        if numSentsOut >= 1:
            break
    
    generatedPromptsList = saveAndPassToImageGen(goodOutputs)







