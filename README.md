
![WhatsApp Image 2024-03-15 at 20 32 44](https://github.com/Asanta4/LLMs---Offensive-or-not-/assets/136238984/d35ee3ec-d43f-4ca0-b021-3592a9d90456)

Report - Final Course Project Advances ML and DL Course
Toxic & Harmful Text Classifier using Traditional NN, Mistral,
and Llama LLMs

By: Raz Graider, Dana Braynin, Sophie Margolis and Ran Asanta



Introduction

In recent years, the rapid expansion of social media platforms and online communities has not
only facilitated greater connectivity and communication but has also revealed a darker side:
hate speech and online harassment.
In this project, we explored the challenge of detecting hate speech using machine learning, NLP
techniques, and LLMs for sentiment analysis. Our focus was on comparing a Keras neural
network model to two advanced Large Language Models (LLMs): Mistral-7b and Llama-2-7b.
The objective was to classify text as offensive or non-offensive (a binary classification task),
leveraging three distinct datasets.
Our approach involved training a Keras neural network on the entire dataset, which comprised
140,218 rows of text collected from various sources, including Wikipedia forums. However, the
LLMs were fine-tuned using only 0.08% of the training data, utilizing LoRa (Low-Rank
Adaptation), a technique for optimizing model performance. Additionally, we explored the
performance of the LLMs with prompt engineering.
The results presented below offer insights into hate speech detection. Through our exploration,
we gained a deeper understanding of the challenges, strengths, and limitations of each model
or approach for this specific task.



Data Preparation

We used 3 data sets we took from Hugging Face 🤗:
1. stormfront_dataset: Contains data related to hate speech.
2. wiki_dataset: Consists of toxic comments from Wikipedia.
3. jigsaw_dataset: Includes multilingual toxicity data from Jigsaw.
We removed duplicated rows from each data set and performed a train-validation-test split on
each one. For the training data, we combined the 3 training data sets and used it for Keras. For
the LLMs we used only 0.08% of it.
Exploratory Data Analysis
By observing the EDA we did, we learned that we have 140,102 rows in our combined train data
set (1.0 - Data train information) and two types of labels: 0 and 1. We discovered that we have
15,834 samples of offensive comments and 124,268 samples of non-offensive comments (1.1 -
Label count).
We also explored the top 20 most common words, with and without stop words, and none of
these words are offensive, which aligns with the fact that our data has much more non-offensive
comments than offensive. Then, we explored the most common words only from the offensive
comments - we can tell that most of them are just hate words and words that indicate racial and
sexual insults (1.2 - top 20 most common from offensive comments). Moreover, we checked the
average length of each type of comment and also checked the top 15 3-grams-words (1.3 - Top
15 3-grams).



Keras NN

Firstly, performed preprocessing on the train data by removing special characters, removing
stop words and performing lemmatization. After that, we also applied the preprocessing on the
validation and test data.
We performed tokenization on all the data sets and afterwards, we used a Keras NN model, and
checked its outcome on our binary classification task. We built an architecture for our Keras
model (2.0 - Keras architecture) and we explored various configurations of NN layers and their
parameters:
1. We tried both MaxPooling1D and AveragePooling1D layers, but the first yielded
better results.
2. We attempted to add a BatchNormalization layer, but it negatively affected the
results.
3. We also determined 0.5 to be the final value for the Dropout regularization layer .We've
done so in order to synchronize all Dropout values through our project, from Keras to
LLM and fine-tuning.
4. Our last layer, we used sigmoid, which is common for binary classification tasks. In
addition, the ReLu activation function is used between the hidden layers.
5. The model was compiled using the binary_crossentropy loss function and the
adam optimizer, which is commonly used for binary classification tasks.
Overall, this architecture was chosen and tuned based on experimentation to achieve optimal
performance for classifying text into offensive or non-offensive categories.
We observe the results of the Keras model in the comparison section in this report.
LLM- Theoretical overview
In this section, we researched about the theoretical overview of the LLM models we use in the
project: Mistral and Llama-2. We gave a theoretical overview on LLMs in general and also did a
comparison table between Mistral-7b and Llama-2-7b in particular (2.1 - Mistral-7b vs.
Llama-2-7b).
For each of the LLM models, we explored prompt engineering and fine-tuning for our task,
investigating the strengths and limitations of each model. In our project, one of the main goals is
to compare the accuracy provided by prompt engineering to that of fine-tuning, as both
strategies play crucial roles in enhancing the performance of AI models.
LLM - Preprocessing
Before moving to the models, we performed preprocessing on the data. We did tokenization for
both the train data for Mistral and train data for Llama-2. Both models use the same tokenization
from Hugging Face’s 🤗 AutoTokenizer class in the Transformers library
(LlamaTokenizer) . We also gave an in depth explanation about the tokenization used and
showed how it works. After that we handled the imbalanced data by assigning higher weight to
the minority class (offensive) and lower weight to the majority class (not offensive). The weights
are proportional to the frequency of the classes.
In this section we also defined the performance metric we will use to compare the models which
is accuracy.



LLM – Prompt Engineering and modeling

In the prompt engineering section, we developed two functions: generate_test_prompt (3.0
- Prompt generate function) and predict (3.1 - Prompt predict function). The
generate_test_prompt function creates a test prompt for sentiment analysis tasks, taking a
comment as input and generating a formatted prompt with instructions for analyzing the
sentiment of the comment. The predict function predicts the sentiment of a comment.
Essentially, for each comment, it generates a prompt asking to analyze the sentiment and
returns the label. We utilized some Hugging Face's 🤗 Transformers functions to streamline
the process, such as pipeline(), which generates text from the language model using the
prompt. At the end of the process, the model gave a classification for the comment: offensive or
not offensive.
In the notebook itself, we dived deeper into the concept of prompt engineering, discussing its
advantages and disadvantages. The results of this section are shown below in the section
comparing the models.



LLM – Fine Tuning

In this section, we experimented with the fine-tuning strategy, which involves training a subset of
parameters of a pre-trained model on a specific dataset to enhance its performance for a
particular task or domain. As models grow larger, full fine-tuning, which entails retraining all the
model's parameters, becomes less feasible due to the required time, cost, and resources.
We delved into Parameter-Efficient Fine-Tuning (PEFT) techniques, which make fine-tuning
more manageable in terms of memory and computational resources. For this project, we
fine-tuned the models using the LoRa (Low-Rank Adaptation) PEFT technique from Hugging
Face's 🤗 PEFT library. Further explanations are provided in the notebook.
The LoRa training arguments provided to both Mistral and Llama were the same, with the
Hugging Face's 🤗 TrainingArguments class that serves as a container for all
hyperparameters and configurations related to the training models. (4.0 - Chosen LoRa
parameters; 4.1 - Lora parameters explain; 4.2 - Chosen TrainingArguments; 4.3 -
TrainingArguments explain).
We trained 0.025% of the parameters in Mistral and 0.032% of the parameters in Llama.
Finally we used this Trainer class WeightedCELossTrainer, which utilizes class weights
previously defined to address the challenge of handling unbalanced data (4.4 - Chosen
WeightedCELossTrainer ; 4.5 - WeightedCELossTrainer explain).
After that, we were ready to run the trained fine-tuned models. We examine their results, as well
as the previous parts, in the next comparison part.



Comparison

For this part we connected to WandB’s reports (5.0 - Train reports) to explore the trainer better
and these are our conclusions (more in depth explanations are in the notebook):

● Runtime
In both Mistral and Llama-2 models, the time taken for training exceeds that of
evaluation. Moreover, it's observed that both training and evaluation times for Llama-2
surpass those of Mistral. This observation reinforces the notion of Llama-2’s greater
hardware demand explained in the theoretical section.

● Train & Evaluation Accuracy
Consistent accuracy across all epochs in the Mistral model were observed. This
prompted us to re-run the training in a separate notebook, yielding identical results.
Despite efforts, no exact explanation for this issue could be found. However, it's
noteworthy that both models achieved evaluation accuracies exceeding 90%.

● GPU Power Usage
Both Mistral and Llama-2 models heavily utilize the GPU, often reaching approximately
100% usage.



In this part we also examined the results from all the tests we ran during the project (5.1 - Test
reports) and these are the conclusions (more in depth explanations are in the notebook):

● Results for Jigsaw Dataset
Mistral-7b Prompt Engineering yielded the highest accuracy (0.9326), while Llama-2-7b
Prompt Engineering and Mistral-7b Fine Tuning showed the lowest accuracies (0.5285).
Surprisingly, a simpler NN built with Keras achieved higher accuracy compared to most
of the LLMs. The weights assigned to each model were adjusted to match the
imbalanced nature of the training data, while this data set is balanced, potentially
contributing to the lower accuracy scores observed in several models.

● Results for Stormfront Dataset
Llama-2-7b Fine Tuning achieved the highest accuracy (0.8939), whereas Keras showed
the lowest accuracy (0.8151). Interestingly, the accuracies of the LLMs were quite similar
while using different approaches.

● Results for Wiki Dataset
Mistral-7b Prompt Engineering recorded the highest accuracy (0.95), while Keras
exhibited the lowest (0.9). Overall, accuracies in this dataset were higher compared to
the other test datasets. This could be attributed to the Wiki dataset having the highest
percentage of usage in the training data compared to others.

● Runtime
Keras demonstrated the shortest runtime (3 minutes), while Mistral-7b Fine Tuning
exhibited the longest runtime (144 minutes). Notably, Fine Tuning models showed
significantly higher runtime compared to Prompt Engineering and Keras models.



We also had general conclusions and questions:

● Identical Results in Llama-2-7b Prompt Engineering and Mistral-7b Fine Tuning
We encountered identical results in Mistral-7b Fine Tuning and Llama-2-7b Prompt
Engineering. It raised concerns about potential information leakage, so we attempted to
isolate the issue by running the model in a separate notebook. It yielded identical results,
so we couldn't find a specific reason for this issue.

● Is Fine Tuning Worth It?
The significantly higher runtime associated with fine tuning, observed in both models,
prompts the question of whether the benefits outweigh the computational costs.
Comparing results between prompt engineering and fine tuning reveals that prompt
engineering often achieves comparable or superior performance, leading to doubts
regarding the use of fine tuning in our mission.

● Prompt vs. Fine Tuning
Analysis of results indicates that prompt engineering and fine tuning often yield similar
performance levels, with prompt engineering occasionally outperforming fine tuning.
Moreover, the practical aspects of working with prompt engineering, such as faster
runtime and user-friendliness due to its simplicity and accessibility, further highlight its
advantages over fine tuning. Additionally, the natural language interface of prompting
facilitates experimentation and feedback, enhancing usability.

● The task influences on model selection
Considering the broader applicability of classification tasks across various domains, the
choice of model becomes crucial. In scenarios where accuracy percentages hold critical
importance, such as in medical or economic contexts, the preference may lean towards
models with faster runtimes that provide marginally higher scores, even if sacrificing
some degree of accuracy. Moreover, it's important to acknowledge situations where
prioritizing accuracy over runtime becomes imperative, despite the associated time
costs. In critical contexts even marginal improvements in accuracy can have substantial
real-world consequences. Hence, careful consideration of both accuracy and runtime
trade-offs is essential in selecting the most suitable model for addressing specific task
requirements and constraints.

● What is the best model?
We see that overall, Mistral emerged as the frontrunner in terms of accuracy, particularly
when leveraged alongside prompt engineering. This outcome aligns with the findings
presented by Mistral AI upon the release of their model, where they demonstrated its
superior performance over different iterations of Llama models, notably outperforming
Llama-2-7b.



Explainability

It is challenging to obtain explainability from LLMs due to their complexity and size. To address
the challenge of the lack of explainability in LLMs, we decided to use prompt engineering. In this
approach, instead of directly interpreting the model's internal mechanisms, we designed a
specific prompt tailored to understand desired behaviors from the model (6.0 - Explaining model
prompt). We constructed a prompt that explicitly asks the model to classify a given sentence as
offensive or not, accompanied by a request for an explanation justifying the model's decision.
We gave the models a comment to see their response ('Black football players are the most
overrated thugs in all of sports!') and created a response function (6.1 - Generate response
function).
We noticed that both models classified the sentence as offensive (6.2 - LLMs explainability).
Both models mentioned the fact that the comment is stereotypical and racist. In comparison to
Mistral, Llama gave an example of what word is racist and explained the historical use of the
word and the group that is talked about. At the end of LLM's answer, both models gave an
insight to how people should behave


![WhatsApp Image 2024-04-08 at 14 00 40](https://github.com/Asanta4/LLMs---Offensive-or-not-/assets/136238984/f8a6c4fd-b193-43da-9410-478dede60d6c)
