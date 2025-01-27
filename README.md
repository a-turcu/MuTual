# Enhancing Multi-Turn Dialogue Reasoning: A Fusion of Datasets

Repository adapted from the paper: [MuTual: A Dataset for Multi-Turn Dialogue Reasoning](https://www.aclweb.org/anthology/2020.acl-main.130/) (ACL2020)

## Abstract
The MuTual challenge was designed as a benchmark for testing the reasoning abilities of chatbots in a dialogue context. One of the baselines defined in the MuTual paper makes use of the BERT model, modified to fit a multiple-choice task. This study reproduces the reported baseline results and attempts to improve them by simultaneously fine-tuning on the additional MMLU dataset. Results show that model performance has the potential to be improved by fine-tuning on additional data that is similar enough but not the same as the downstream task data.

Authors: Alexandru Turcu, Bogdan Palfi, Darie Petcu, Marco Gallo




# Example
Example of the MuTual data
<img src="./readme/construct.png" width="1000" >

# Data statistics


|                               |                                             MuTual |                                    MMLU [all, auxiliary_train] |
|-------------------------------|----------------------------------------------------|--------------------------------------------------------------|
| Context-Response Pairs        |                                               8860 |                                                              |
| #Avg. Turns per Dialogue      |                                               4.73 |                                                              |
| #Avg. Words per Utterance     |                                              19.57 |                                                              |
| Vocabulary Size (Context)     |                                               8809 |                                                       140304 |
| Vocabulary Size (Response)    |                                               8943 |                                                        53832 |
| Vocabulary Size               |                                              11343 |                                                       150168 |
| # Original Dialogues          |                                               6371 |                                                              |
| # Original Questions          |                                              11323 |                                                              |
| # Response Candidates         |                                                  4 |                                                            4 |
| Avg. context length           |                                             93.33  |                                                       296.18 |
| Avg. Context length quantiles | [  8,  34,  42,  49,  58,  79, 111, 141, 184, 408] | [   1,  104,  204,  261,  302,  332,  358,  386,  428, 1411] |


# Data template
```data/mutual_plus/train```, ```data/mutual_plus/dev``` and ```data/mutual_plus/test``` are the training, development and test sets of MuTual Plus. After loading each file, you will get a dictionary. The format of them is as follows:

```
{"answers": "B",
"options": ["m : so you come to manchester just for watching a concert , do n't you ?", "m : i really want to say that your performance in manchester must will be great !", "m : you come to manchester specially for this friend , so your friendship must be very deep .", "m : is this your first performance in manchester ? i remember you never sang at a high school concert ."],
"article": "m : hi , della . how long are you going to stay here ? f : only 4 days . i know that 's not long enough , but i have to go to london after the concert here at the weekend . m : i 'm looking forward to that concert very much . can you tell us where you sing in public for the first time ? f : hmm ... at my high school concert , my legs shook uncontrollably and i almost fell . m : i do n't believe that . della , have you been to any clubs in manchester ? f : no , i have n't . but my boyfriend and i are going out this evening . we know manchester has got some great clubs and tomorrow will go to some bars .",
"id": "dev_1"}
```

```data/mmlu/auxiliary_train``` is the train set of MMLU. The format is similar to MuTual.
```
{
  "question": "The best evidence that two land areas were once connected is the discovery that both land masses",
  "choices": ["have the same climate.", "are in the same stage of succession.", "exist along the same line of longitude.", "have similar types of rocks and fossils."],
  "answer": "D"
}
```
The given code from the ```mmlu_utils.py``` file will automatically download the dataset and change the "question", "choices" and "answer" fields to match that of MuTual into "article", "options" and "answers", respectively. An id will also be added.

``` options ``` is a list of four candidates' response.

``` article ```  is the context. ```f``` and ```m``` indicate female and male, respectively.

```answers``` is the correct answer. Noted that we do not realease the correct answer on test set.

# How to run the code

### Create and activate environment
```sh
conda env create -f env.yml
conda activate dl4nlp
```

### Save MMLU Data
The MuTual data is already available in the repository.
```sh
python baseline/multi_choice/mmlu_utils.py
```

### Run classic (original) version with speaker characters removed
To not remove the speaker characters, simply delete the "--remove_speakers" flag and change the "--output_dir" flag
```sh
python baseline/multi_choice/run_multiple_choice.py --num_train_epochs 10 --data_dir "data/mutual_plus" --model_type "bert" --model_name_or_path "bert-base-uncased" --task_name "mutual" --output_dir "output/bert/classic/no-speakers" --do_train --evaluate_during_training --do_lower_case --overwrite_output_dir --overwrite_cache --remove_speakers
```

### Run Random Mix
To run the experiment with random sampling from MMLU, use the "random_mix" option for the "--train_mode" flag.
```sh
python baseline/multi_choice/run_multiple_choice.py --train_mode "random_mix" --percentage 0.04 --data_dir "data/mutual_plus" --model_type "bert" --model_name_or_path "bert-base-uncased" --task_name "mutual" --output_dir "output/bert/random_mix" --do_train --evaluate_during_training --do_lower_case --overwrite_output_dir --overwrite_cache --remove_speakers
```

### Run Informed Mix
To run the experiment with euclidean-distance-based sampling from MMLU, use the "embeddings_mix" option for the "--train_mode" flag.
```sh
python baseline/multi_choice/run_multiple_choice.py --train_mode "embeddings_mix" --percentage 0.04 --data_dir "data/mutual_plus" --model_type "bert" --model_name_or_path "bert-base-uncased" --task_name "mutual" --output_dir "output/bert/inf_mix" --do_train --evaluate_during_training --do_lower_case --overwrite_output_dir --overwrite_cache --remove_speakers
```

### Select amount of MMLU to use for training alongside the MuTual data
To adjust the percentage of MMLU sampled to use in training, change the "--percentage" flag value to the desired percentage, as a value between [0,1].
For example, 0.1 will sample 10% of MMLU. **Make sure** the "--train_mode" argument is set to either "random_mix" or "embeddings_mix".
```sh
python baseline/multi_choice/run_multiple_choice.py --train_mode "embeddings_mix" --percentage 0.1 --data_dir "data/mutual_plus" --model_type "bert" --model_name_or_path "bert-base-uncased" --task_name "mutual" --output_dir "output/bert/inf_mix" --do_train --evaluate_during_training --do_lower_case --overwrite_output_dir --overwrite_cache --remove_speakers
```
