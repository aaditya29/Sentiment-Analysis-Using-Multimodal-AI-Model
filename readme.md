# Sentiment Analysis Using Multimodal AI Model

## Introduction

In this project we are going to use PyTorch to create and train a multimodal AI model from scratch.<br>
A video will be given as input into the model, which will then predict its feeling and emotion. The model will be trained using characteristics such multimodal fusion, emotion and sentiment classification with text, video, and audio encoding. <br>Following the model's deployment and training, we will create a Software As A Service (SaaS) website interface that allows us to utilise API to perform inference on their own videos. We will control the monthly quotas that users have and set up the deployment model's invocation using SageMaker Endpoints. The SaaS is based on the T3 Stack and will be developed using technologies like Next.js, React, Tailwind, and Auth.js.

## What Is Model About?

We are going to do `Sentiment Analysis of Video` using Multimodal EmotionLines Dataset (MELD). <br>
MELD contains the same dialogue instances available in EmotionLines, but it also encompasses audio and visual modality along with text. MELD has more than 1400 dialogues and 13000 utterances from <b>Friends TV series.</b><br>
Multiple speakers participated in the dialogues. Each utterance in a dialogue has been labeled by any of these seven emotions -- Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear. MELD also has sentiment (positive, negative and neutral) annotation for each utterance.
